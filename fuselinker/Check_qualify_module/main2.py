import argparse
import os
import numpy as np
import pandas as pd
import torch
import pickle

import task_setup  # noqa: F401  (đưa thư mục này + ../common lên sys.path)
from paths import (default_checkpoint, ensure_parent,
                   resolve_data_dir, resolve_path)

import myutils
from calc_auroc import plot_roc_curve, roc_curve_path_from_checkpoint
from model_base2 import LinkPredict, compute_ppr_sparse
from data_loader import Data


def main(args):
    # --data nhận tên dataset (suppkg, hetionet, ...) và luôn resolve về
    # fuselinker/<tên>, không phụ thuộc thư mục đang đứng.
    args.data = resolve_data_dir(args.data)
    args.model_state_file = ensure_parent(resolve_path(args.model_state_file))
    if args.roc_save_path:
        args.roc_save_path = ensure_parent(resolve_path(args.roc_save_path))
    print(f"Data: {args.data}")
    print(f"Checkpoint: {args.model_state_file}")
    train_path = f'{args.data}/train.tsv'
    valid_path = f'{args.data}/valid.tsv'
    test_path = f'{args.data}/test.tsv'
    text_embedding_path = f'{args.data}/{args.text_embedding_file}'
    knowledge_embedding_path = f'{args.data}/{args.knowledge_embedding_file}'
    freeze = args.freeze

    train = pd.read_csv(train_path, sep='\t', header=None)
    valid = pd.read_csv(valid_path, sep='\t', header=None)
    test = pd.read_csv(test_path, sep='\t', header=None)
    graph = pd.concat([train, valid, test]) 

    print("Loading Pretrained Embeddings files...")
    try:
        text_embeddings = np.load(text_embedding_path)
        print("Loaded Text Embeddings file successfully!")
    except:
        text_embeddings = None
        print("Failed to Text Embeddings file, random embeddings will be  created.")

    try:
        ontology_embeddings = np.load(knowledge_embedding_path)
        print("Loaded Domain Knowledge Embeddings file successfully!")
    except:
        ontology_embeddings = None
        print("Failed to load Domain Knowledge Embeddings file, random embeddings will be  created.")

    print(f"w: {args.w}")
    
    print("Data Processing...")
    knowledge_graph = Data(graph, train, valid, test)
    num_nodes, num_rels, num_edges = knowledge_graph.get_stats()
    print('# entities:', num_nodes)
    print('# relations:', num_rels)
    print('# edges:', num_edges)
    #Ghi knowledge_graph.relation2index vào file relation2index.pkl
    with open(f'{args.data}/relation2index.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.relation2index, file)
    with open(f'{args.data}/index2relation.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.index2relation, file)
    with open(f'{args.data}/entity2index.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.entity2index, file)
    with open(f'{args.data}/index2entity.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.index2entity, file)

    train_data_np = knowledge_graph.train_data
    valid_data_np = knowledge_graph.valid_data
    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data

    train_data = torch.LongTensor(train_data_np)
    valid_data = torch.LongTensor(valid_data_np)
    test_data = torch.LongTensor(test_data_np)
    total_data = torch.LongTensor(total_data_np)

    model = LinkPredict(num_nodes,
                        args.n_hidden,
                        num_rels,
                        num_bases=args.num_bases,
                        num_hidden_layers=args.num_hidden_layers,
                        dropout=args.dropout,
                        use_cuda=args.use_cuda,
                        regularization_param=args.reg_param,
                        pretrained_text_embeddings=text_embeddings,
                        pretrained_domain_embeddings=ontology_embeddings,
                        freeze=freeze,
                        w=args.w,
                        use_ppr=args.use_ppr,
                        ppr_num_layers=args.ppr_num_layers)


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = model.to(device)
    print(device)

    # build train graph
    train_graph, train_rel, train_norm = myutils.build_graph(num_nodes, num_rels, train_data_np)
    train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)

    # build test graph
    test_graph, test_rel, test_norm = myutils.build_graph(num_nodes, num_rels, test_data_np)
    test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1, 1)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    test_rel = torch.from_numpy(test_rel)
    test_norm = myutils.node_norm_2_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

    # build adj list and calculate degrees for sampling
    adj_list = myutils.get_adj(num_nodes, train_data_np)  # degrees

    # ========================================================
    # Generating Auxiliary Network (PPR) - theo HGDC
    # ========================================================
    if args.use_ppr:
        print("Building auxiliary PPR network...")
        ppr_graph, ppr_edge_weight = compute_ppr_sparse(
            num_nodes, train_data_np,
            c=args.ppr_c, epsilon=args.ppr_eps,
            use_iterative=args.ppr_iterative,
            num_iter=args.ppr_iter_num,
        )
        ppr_graph = ppr_graph.to(device)
        ppr_edge_weight = ppr_edge_weight.to(device)
        all_node_ids = torch.arange(num_nodes, dtype=torch.long).view(-1, 1).to(device)
        print(f"Gppr: {ppr_graph.number_of_nodes()} nodes, "
              f"{ppr_graph.number_of_edges()} edges "
              f"(c={args.ppr_c}, epsilon={args.ppr_eps}).")
    else:
        ppr_graph = None
        ppr_edge_weight = None
        all_node_ids = None

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # training loop
    print("Start training...")

    # epoch is step
    for iteration in range(1, 1 + args.iterations):
        model.train()

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            myutils.generate_sampled_graph_and_labels(
                train_data_np, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, train_deg, args.negative_sample,
                args.edge_sampler)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = myutils.node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

        # Load on device
        g = g.to(device)
        node_id = node_id.to(device)
        edge_type = edge_type.to(device)
        edge_norm = edge_norm.to(device)
        data = data.to(device)
        labels = labels.to(device)

        embed = model(g, node_id, edge_type, edge_norm,
                      ppr_graph=ppr_graph,
                      ppr_edge_weight=ppr_edge_weight,
                      all_node_ids=all_node_ids)
        loss = model.get_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()

        if iteration % args.evaluate_every == 0:
            print("Epoch {} | Loss {:.5f}".format(iteration, loss.item()))

        optimizer.zero_grad()

    torch.save({'state_dict': model.state_dict(), 'iteration': iteration}, ensure_parent(args.model_state_file))

    print("Evaluating...")
    model.eval()
    test_data = torch.LongTensor(test_data_np)
    total_data = torch.LongTensor(total_data)

    # Dùng đồ thị train để tính embedding (chuẩn link prediction), không dùng test graph
    eval_graph = train_graph.to(device)
    eval_node_id = test_node_id.to(device)
    eval_rel = torch.from_numpy(train_rel).to(device)
    eval_norm = myutils.node_norm_2_edge_norm(
        train_graph, torch.from_numpy(train_norm).view(-1, 1)
    ).to(device)
    test_data = test_data.to(device)
    total_data = total_data.to(device)

    output = model(eval_graph, eval_node_id, eval_rel, eval_norm,
                   ppr_graph=ppr_graph,
                   ppr_edge_weight=ppr_edge_weight,
                   all_node_ids=all_node_ids)

    import time
    old_time = time.time()

    hits = [1, 3, 10]
    mr, mrr, hits_dict = myutils.calc_mrr(output, model.relation_weights, test_data,
                                            total_data,
                                            batch_size=args.eval_batch_size, neg_sample_size_eval=args.neg_sample_size_eval,
                                            hits=hits, eval_p=args.eval_protocol)

    new_time = time.time()
    print(new_time - old_time)

    print(f"MR: {mr:.6f}")
    print(f"MRR: {mrr:.6f}")
    for key, value in hits_dict.items():
        print(f"Hits @ {key} = {value:.6f}")

    auroc, fpr, tpr, _, _ = model.evaluate_auroc(
        output, test_data, total_data, seed=42,
    )
    print(f"AUROC: {auroc:.6f}")

    roc_save_path = args.roc_save_path or ensure_parent(
        resolve_path(roc_curve_path_from_checkpoint(args.model_state_file)))
    plot_roc_curve(fpr, tpr, auroc, save_path=roc_save_path)
    print(f"ROC curve saved: {roc_save_path}")

    print("Training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser For Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="data",
        default="suppkg",
        help="Tên dataset dưới fuselinker/ (suppkg, hetionet, ...) hoặc đường dẫn",
    )
    # file này chứa text embedding cho từng node
    parser.add_argument(
        "--text_embedding_file",
        dest="text_embedding_file",
        default="bert_pretrained_embeddings_768.npy",
        help="Path of text embedding for each node",
    )
    # file này chứa domain-knowledge embedding cho từng node
    parser.add_argument(
        "--knowledge_embedding_file",
        dest="knowledge_embedding_file",
        default="poincare_embeddings.npy",
        help="Path of domain knowledge embedding for each node",
    )
    
    # biến freeze quyết định xem text_embedding và domain_embedding có được học hay ko?
    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze text embedding and domain knowledge or not"
    )
    
    # biến w quyết định tỉ lệ trộn text_embedding và domain_embedding
    parser.add_argument(
        "--w",
        dest="w",
        type=float,
        default=0.5,
        help="The weight for fusing embedings",
    )
    #biến n_hidden: Kích thước vector dùng trong RGCN và autoencoder giảm chiều embedding.Đây cũng là chiều relation embedding
    parser.add_argument(
        "--n_hidden",
        dest="n_hidden",
        type=int,
        default=200,
        help="Dimensions of the hidden layer",
    )

    parser.add_argument(
        "--num_bases",
        dest="num_bases",
        type=int,
        default=20,
        help="Number of basis relation vectors to use",
    )
    #Số lớp RGC hidden -> nhận thông tin từ hàng xóm xa(tỉnh táo vì dễ over-smoothing)
    parser.add_argument(
        "--num_hidden_layers",
        dest="num_hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers",
    )
    #Tỉ lệ dropout trong RGCN và autoencoder giảm chiều embedding
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=0.2,
        help="Number of hidden layers",
    )

    parser.add_argument(
        "--use_cuda",
        dest="use_cuda",
        type=bool,
        default=True,
        help="GPU",
    )
# hệ số regularization loss trong hàm loss 
    parser.add_argument(
        "--reg_param",
        dest="reg_param",
        type=float,
        default=0.01,
        help="GPU",
    )


    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=1,
        help="Number of iterations (iterations = epochs * (datasize / batchsize))",
    )
    #Chu kỳ log, in ra loss 
    parser.add_argument(
        "--evaluate_every",
        dest="evaluate_every",
        type=int,
        default=4000,
    )
    #Tham số learning rate
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    #Tham số batchsize
    parser.add_argument(
        "--graph_batch_size",
        dest="graph_batch_size",
        type=int,
        default=250,
    )
    #Tỉ lệ cạnh trong subgprah dùng làm positive triples, phần còn lại làm message-passing(dùng lan truyền thông tin trong RGCN)
    parser.add_argument(
        "--graph_split_size",
        dest="graph_split_size",
        type=float,
        default=0.5,
    )

# SỐ mẫu negative sample khi đánh giá 
    parser.add_argument(
        "--negative_sample",
        dest="negative_sample",
        type=int,
        default=20,
    )
    
    #Kiếu lấy sample: uniform lấy ngẫu nhiên 
    parser.add_argument(
        "--edge_sampler",
        dest="edge_sampler",
        default="uniform",
    )

    parser.add_argument(
        "--grad_norm",
        dest="grad_norm",
        type=float,
        default=1.0,
    )

# Số tiple test xử lý đồng thời khi tính MRR, Hits@K
    parser.add_argument(
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=50,
    )

#Số mẫu negative sample khi đánh giá 
    parser.add_argument(
        "--neg_sample_size_eval",
        dest="neg_sample_size_eval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--eval_protocol",
        dest="eval_protocol",
        default="filtered",
    )

    parser.add_argument(
        "--model_state_file",
        dest="model_state_file",
        default=default_checkpoint("suppkg", "bert", "orig_main2"),
        help="Đường dẫn checkpoint; path tương đối resolve dưới fuselinker/.",
    )

    parser.add_argument(
        "--roc_save_path",
        dest="roc_save_path",
        default=None,
        help="ROC plot path. Default: same folder as model_state_file, file roc_curve.png",
    )

    # ======= Auxiliary PPR network =======
    parser.add_argument(
        "--use_ppr", dest="use_ppr",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="Enable auxiliary PPR network branch (HGDC-style).",
    )
    parser.add_argument(
        "--ppr_c", dest="ppr_c", type=float, default=0.15,
        help="Damping factor c in PPR (teleport prob = 1 - c).",
    )
    parser.add_argument(
        "--ppr_eps", dest="ppr_eps", type=float, default=1e-4,
        help="Truncation threshold epsilon for sparsifying S.",
    )
    parser.add_argument(
        "--ppr_num_layers", dest="ppr_num_layers", type=int, default=2,
        help="Number of PPR-GCN layers.",
    )
    parser.add_argument(
        "--ppr_iterative", dest="ppr_iterative",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
        help="Use iterative PPR (saves RAM) instead of dense matrix inverse.",
    )
    parser.add_argument(
        "--ppr_iter_num", dest="ppr_iter_num", type=int, default=50,
        help="Number of iterations if --ppr_iterative True.",
    )

    args = parser.parse_args()

    main(args)