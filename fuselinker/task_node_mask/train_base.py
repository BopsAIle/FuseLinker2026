"""
Nhiệm vụ 1 — DistMult / dự đoán entity bị mask (encoder BASE: model.py).

Che head hoặc tail của triple, chấm điểm DistMult trên subgraph đã sample.
"""
import task_setup  # noqa: F401
from paths import default_checkpoint, ensure_parent, resolve_data_dir, resolve_path

import argparse
import pickle

import numpy as np
import pandas as pd
import torch

import myutils
from calc_auroc import plot_roc_curve, roc_curve_path_from_checkpoint
from model import LinkPredict
from data_loader import Data


def main(args):
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
    except Exception:
        text_embeddings = None
        print("Failed to Text Embeddings file, random embeddings will be  created.")

    try:
        ontology_embeddings = np.load(knowledge_embedding_path)
        print("Loaded Domain Knowledge Embeddings file successfully!")
    except Exception:
        ontology_embeddings = None
        print("Failed to load Domain Knowledge Embeddings file, random embeddings will be  created.")

    print(f"w: {args.w}")

    print("Data Processing...")
    knowledge_graph = Data(graph, train, valid, test)
    num_nodes, num_rels, num_edges = knowledge_graph.get_stats()
    print('# entities:', num_nodes)
    print('# relations:', num_rels)
    print('# edges:', num_edges)
    with open(f'{args.data}/relation2index.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.relation2index, file)
    with open(f'{args.data}/index2relation.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.index2relation, file)
    with open(f'{args.data}/entity2index.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.entity2index, file)
    with open(f'{args.data}/index2entity.pkl', 'wb') as file:
        pickle.dump(knowledge_graph.index2entity, file)

    train_data_np = knowledge_graph.train_data
    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data

    model = LinkPredict(
        num_nodes,
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
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model = model.to(device)
    print(device)
    print("Task 1 DistMult | Encoder: model.py (base RGCN)")

    train_graph, train_rel, train_norm = myutils.build_graph(num_nodes, num_rels, train_data_np)
    train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)

    # FuseLinker gốc (main.py): lúc TEST encode trên graph dựng từ test.tsv.
    test_graph, test_rel, test_norm = myutils.build_graph(
        num_nodes, num_rels, test_data_np
    )
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    adj_list = myutils.get_adj(num_nodes, train_data_np)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Start training (masked-entity DistMult)...")

    for iteration in range(1, 1 + args.iterations):
        model.train()

        g, node_id, edge_type, node_norm, data, labels = \
            myutils.generate_sampled_graph_and_labels(
                train_data_np, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, train_deg, args.negative_sample,
                args.edge_sampler)

        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = myutils.node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)

        g = g.to(device)
        node_id = node_id.to(device)
        edge_type = edge_type.to(device)
        edge_norm = edge_norm.to(device)
        data = data.to(device)
        labels = labels.to(device)

        embed = model(g, node_id, edge_type, edge_norm)
        loss = model.get_loss(g, embed, data, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()

        if iteration % args.evaluate_every == 0:
            print("Epoch {} | Loss {:.5f}".format(iteration, loss.item()))

        optimizer.zero_grad()

    torch.save(
        {'state_dict': model.state_dict(), 'iteration': iteration},
        ensure_parent(args.model_state_file),
    )

    print(
        "Evaluating TEST: encode on test_graph "
        "(FuseLinker original main.py protocol)."
    )
    model.eval()
    test_data = torch.LongTensor(test_data_np)
    total_data = torch.LongTensor(total_data_np)

    eval_graph = test_graph.to(device)
    eval_node_id = test_node_id.to(device)
    eval_rel = torch.from_numpy(test_rel).to(device)
    eval_norm = myutils.node_norm_2_edge_norm(
        test_graph, torch.from_numpy(test_norm).view(-1, 1)
    ).to(device)
    test_data = test_data.to(device)
    total_data = total_data.to(device)

    output = model(eval_graph, eval_node_id, eval_rel, eval_norm)

    import time
    old_time = time.time()

    hits = [1, 3, 10]
    mr, mrr, hits_dict = myutils.calc_mrr(
        output, model.relation_weights, test_data,
        total_data,
        batch_size=args.eval_batch_size,
        neg_sample_size_eval=args.neg_sample_size_eval,
        hits=hits, eval_p=args.eval_protocol,
    )

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
        resolve_path(roc_curve_path_from_checkpoint(args.model_state_file))
    )
    plot_roc_curve(fpr, tpr, auroc, save_path=roc_save_path)
    print(f"ROC curve saved: {roc_save_path}")

    print("Training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 1 base: DistMult masked-entity prediction (model.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data", dest="data", default="hetionet",
        help="Dataset folder name under fuselinker/ (e.g. hetionet) or a path",
    )
    parser.add_argument(
        "--text_embedding_file", dest="text_embedding_file",
        default="pubmedbert_embeddings_768.npy",
        help="Path of text embedding for each node",
    )
    parser.add_argument(
        "--knowledge_embedding_file", dest="knowledge_embedding_file",
        default="poincare_embeddings.npy",
        help="Path of domain knowledge embedding for each node",
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze text embedding and domain knowledge or not",
    )
    parser.add_argument(
        "--w", dest="w", type=float, default=0.5,
        help="The weight for fusing embedings",
    )
    parser.add_argument(
        "--n_hidden", dest="n_hidden", type=int, default=200,
        help="Dimensions of the hidden layer",
    )
    parser.add_argument(
        "--num_bases", dest="num_bases", type=int, default=20,
        help="Number of basis relation vectors to use",
    )
    parser.add_argument(
        "--num_hidden_layers", dest="num_hidden_layers", type=int, default=1,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--dropout", dest="dropout", type=float, default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--use_cuda", dest="use_cuda", type=bool, default=True, help="GPU",
    )
    parser.add_argument(
        "--reg_param", dest="reg_param", type=float, default=0.01,
        help="Regularization param",
    )
    parser.add_argument(
        "--iterations", dest="iterations", type=int, default=1,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--evaluate_every", dest="evaluate_every", type=int, default=4000,
    )
    parser.add_argument(
        "--lr", dest="lr", type=float, default=0.001, help="Learning rate",
    )
    parser.add_argument(
        "--graph_batch_size", dest="graph_batch_size", type=int, default=250,
    )
    parser.add_argument(
        "--graph_split_size", dest="graph_split_size", type=float, default=0.5,
        help="Fraction of sampled edges kept for message-passing; rest are DistMult positives",
    )
    parser.add_argument(
        "--negative_sample", dest="negative_sample", type=int, default=20,
    )
    parser.add_argument(
        "--edge_sampler", dest="edge_sampler", default="uniform",
    )
    parser.add_argument(
        "--grad_norm", dest="grad_norm", type=float, default=1.0,
    )
    parser.add_argument(
        "--eval_batch_size", dest="eval_batch_size", type=int, default=50,
    )
    parser.add_argument(
        "--neg_sample_size_eval", dest="neg_sample_size_eval", type=int, default=20,
    )
    parser.add_argument(
        "--eval_protocol", dest="eval_protocol", default="filtered",
    )
    parser.add_argument(
        "--model_state_file", dest="model_state_file",
        default=default_checkpoint("hetionet", "pubmedbert", "model"),
        help="Checkpoint path (relative paths resolve under fuselinker/).",
    )
    parser.add_argument(
        "--roc_save_path", dest="roc_save_path", default=None,
        help="ROC plot path. Default: same folder as model_state_file, file roc_curve.png",
    )

    args = parser.parse_args()
    main(args)
