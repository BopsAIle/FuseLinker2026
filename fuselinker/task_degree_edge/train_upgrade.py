"""
Nhiệm vụ 2 — SSL upgrade: dự đoán bậc node + loại cạnh bị mask (encoder model_base4.py).

Cùng nhiệm vụ và độ đo với train_base.py (ssl_eval.py):
    - Mask cạnh khỏi message-passing graph; predict trên cạnh mask + negative.
    - loss_edge: phân loại cạnh theo loại quan hệ + 1 lớp "negative/fake edge".
    - loss_node: phân loại bậc (degree) của node vào các bucket.
    loss = loss_edge + loss_node   (thuần SSL, không dùng DistMult)

Độ đo log mỗi epoch (tách 2 task):
  loss, auc, aupr, f1, bacc (+ bản node)
"""
import task_setup  # noqa: F401
from paths import default_checkpoint, ensure_parent, resolve_data_dir, resolve_path

import argparse
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import myutils
from lukepi_graph import generate_lukepi_masked_graph_and_labels
from model_base4 import (
    LinkPredict,
    EdgeTypeClassifier,
    DegreeClassifier,
    compute_ppr_sparse,
)
from data_loader import Data
from ssl_eval import (
    UncertaintyWeighter,
    add_ssl_log,
    average_ssl_log,
    batch_ssl_log,
    build_degree_labels,
    build_ssl_checkpoint,
    empty_ssl_log,
    encode_full_train_graph,
    eval_ssl_log,
    evaluate_ssl_degree,
    evaluate_ssl_edge,
    final_ckpt_path,
    format_ssl_log,
)


def main(args):
    args.data = resolve_data_dir(args.data)
    args.ssl_model_state_file = ensure_parent(resolve_path(args.ssl_model_state_file))
    print(f"Data: {args.data}")
    print(f"SSL checkpoint: {args.ssl_model_state_file}")
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

    # Pretrained Embedding.from_pretrained sizes the table from .npy rows,
    # while training (especially --use_ppr) looks up ids in [0, num_nodes).
    # Mismatch → CUDA: indexSelectLargeIndex / srcIndex < srcSelectDimSize.
    for _name, _arr in (("text", text_embeddings), ("domain", ontology_embeddings)):
        if _arr is None:
            continue
        if _arr.ndim != 2:
            raise ValueError(f"{_name} embeddings must be 2D, got shape {_arr.shape}")
        if _arr.shape[0] != num_nodes:
            raise ValueError(
                f"{_name} embeddings rows={_arr.shape[0]} != num_nodes={num_nodes}. "
                f"Rebuild/align .npy so row i = entity with id i in entity2index "
                f"(first-appearance order in train+valid+test, not sorted names)."
            )
        print(f"{_name} embeddings aligned: {_arr.shape}")

    # Ghi dictionary index để tái sử dụng downstream.
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
    eval_neg_rate = (
        args.negative_sample if args.eval_neg_rate is None else args.eval_neg_rate
    )
    eval_rng = np.random.RandomState(42)

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
    print("Encoder: model_base4.py (RGCN + optional PPR)")

    # SSL heads (LukePi-style): dự đoán loại cạnh và bậc node.
    edge_head = EdgeTypeClassifier(args.n_hidden, num_rels + 1, dropout=args.dropout).to(device)
    degree_head = DegreeClassifier(args.n_hidden, args.num_degree_bins, dropout=args.dropout).to(device)

    # build train graph
    train_graph, train_rel, train_norm = myutils.build_graph(num_nodes, num_rels, train_data_np)
    # Mảng chứa bậc của mỗi node
    train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)

    # build adj list and calculate degrees for sampling
    adj_list = myutils.get_adj(num_nodes, train_data_np)  # degrees

    # Nhãn bucket bậc (tính 1 lần từ bậc toàn đồ thị train).
    degree_label_full, degree_bin_edges = build_degree_labels(
        train_deg, args.num_degree_bins, strategy=args.degree_bin_strategy
    )
    degree_label_full = degree_label_full.to(device)
    unique_bins = torch.unique(degree_label_full).tolist()
    print(f"Degree bins: strategy={args.degree_bin_strategy}, K={args.num_degree_bins}, "
          f"num_used_bins={len(unique_bins)}, edges={degree_bin_edges}")

    # Class-weight cho degree: bin log2 rất mất cân bằng (đa số node bậc thấp).
    # CE không trọng số -> model thiên về bucket đa số -> kéo tụt bacc/f1/aupr (đúng
    # các chỉ số degree đang thua base). Trọng số ~ 1/sqrt(freq), chuẩn hóa mean ~ 1.
    deg_class_counts = torch.bincount(
        degree_label_full, minlength=args.num_degree_bins
    ).float().clamp(min=1.0)
    deg_class_weights = 1.0 / torch.sqrt(deg_class_counts)
    deg_class_weights = deg_class_weights * (
        args.num_degree_bins / deg_class_weights.sum()
    )
    deg_class_weights = deg_class_weights.to(device)
    print(
        f"Degree class weights: min={deg_class_weights.min():.3f}, "
        f"max={deg_class_weights.max():.3f}, degree_on_full={args.degree_on_full}"
    )

    # ========================================================
    # Generating Auxiliary Network (PPR) - theo HGDC
    # ========================================================
    if args.use_ppr:
        print(f"Building auxiliary PPR network on {device}...")
        ppr_graph, ppr_edge_weight = compute_ppr_sparse(
            num_nodes, train_data_np,
            c=args.ppr_c, epsilon=args.ppr_eps,
            use_iterative=args.ppr_iterative,
            num_iter=args.ppr_iter_num,
            device=device,
            batch_size=args.ppr_batch_size,
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

    # Cân bằng 2 task: học trọng số (uncertainty) hoặc lambda tĩnh.
    if args.learn_loss_weights:
        loss_weighter = UncertaintyWeighter().to(device)
        print(
            "Loss weighting: learned uncertainty (Kendall) — "
            "lambda_edge/lambda_node tự học."
        )
    else:
        loss_weighter = None
        print(f"Loss weighting: static — lambda_degree={args.lambda_degree}")

    # optimizer: gộp encoder + 2 head (+ loss_weighter) vào 1 optimizer.
    params = (
        list(model.parameters())
        + list(edge_head.parameters())
        + list(degree_head.parameters())
    )
    if loss_weighter is not None:
        params += list(loss_weighter.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # training loop
    print("Start SSL pretraining (LukePi edge-mask + degree)...")
    print(
        f"Train log every {args.evaluate_every} iters "
        f"(loss, auc, aupr, f1, bacc + bản node); "
        f"valid checkpoint every {args.evaluate_every} "
        f"(eval_neg_rate={eval_neg_rate}, eval_max_triples={args.eval_max_triples})"
    )

    best_valid_loss_edge = float("inf")
    best_iter = None
    last_iteration = 0
    train_log_sum = empty_ssl_log()
    n_since_log = 0

    for iteration in range(1, 1 + args.iterations):
        last_iteration = iteration
        model.train()
        edge_head.train()
        degree_head.train()

        # LukePi-style: sample subgraph -> mask cạnh khỏi g -> predict mask + neg.
        # data: [N, 3] (src, rel, dst) local id; edge_labels: [N] lớp [0, num_rels].
        # Cứ mỗi 1 vòng lặp, ta sẽ sinh ra 1 subgraph mới
        g, node_id, edge_type, node_norm, data, edge_labels = \
            generate_lukepi_masked_graph_and_labels(
                train_data_np, args.graph_batch_size, args.mask_rate,
                num_rels, adj_list, train_deg, args.negative_sample,
                args.edge_sampler)

        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = myutils.node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data = torch.from_numpy(data)
        edge_labels = torch.from_numpy(edge_labels).long()

        # Load on device
        g = g.to(device)
        node_id = node_id.to(device)
        edge_type = edge_type.to(device)
        edge_norm = edge_norm.to(device)
        data = data.to(device)
        edge_labels = edge_labels.to(device)

        # Encode trên đồ thị ĐÃ BỎ cạnh mask (pipeline model_base4.LinkPredict).
        embed = model(g, node_id, edge_type, edge_norm,
                      ppr_graph=ppr_graph,
                      ppr_edge_weight=ppr_edge_weight,
                      all_node_ids=all_node_ids)

        # ---- edge-type loss (chỉ trên mask + negative) ----
        edge_logits = edge_head(embed[data[:, 0]], embed[data[:, 2]])  # [N, num_rels+1]
        loss_edge = F.cross_entropy(edge_logits, edge_labels)

        #---- degree loss ----
        # Task degree KHÔNG cần đồ thị mask (mask chỉ phục vụ task edge).
        # --degree_on_full: encode TOÀN BỘ đồ thị train để
        #   (1) phủ đều mọi node thay vì thiên về hub,
        #   (2) tận dụng nhánh PPR full-graph nhất quán lúc train == lúc TEST,
        #   (3) khớp phân phối với evaluate_ssl_degree (nhãn bậc từ train_deg).
        # Ngược lại: giữ hành vi cũ (giám sát trên node của subgraph).
        if args.degree_on_full:
            deg_embed = encode_full_train_graph(
                model, train_graph, train_rel, train_norm, num_nodes, device,
                ppr_graph=ppr_graph,
                ppr_edge_weight=ppr_edge_weight,
                all_node_ids=all_node_ids,
            )
            deg_labels = degree_label_full
        else:
            deg_embed = embed
            deg_labels = degree_label_full[node_id.squeeze()]  # original ids -> bucket
        deg_logits = degree_head(deg_embed)
        loss_node = F.cross_entropy(deg_logits, deg_labels, weight=deg_class_weights)

        if loss_weighter is not None:
            loss = loss_weighter(loss_edge, loss_node)
            w_edge, w_node = loss_weighter.weights()
        else:
            loss = loss_edge + args.lambda_degree * loss_node
            w_edge, w_node = 1.0, args.lambda_degree
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, args.grad_norm)  # clip gradients
        optimizer.step()

        add_ssl_log(
            train_log_sum,
            batch_ssl_log(
                loss, loss_edge, loss_node,
                edge_logits, edge_labels, deg_logits, deg_labels,
                w_edge, w_node,
            ),
        )
        n_since_log += 1

        if iteration % args.evaluate_every == 0:
            print(format_ssl_log(
                "Training average",
                iteration,
                average_ssl_log(train_log_sum, n_since_log),
            ))
            train_log_sum = empty_ssl_log()
            n_since_log = 0

            model.eval()
            edge_head.eval()
            degree_head.eval()

            full_embed = encode_full_train_graph(
                model, train_graph, train_rel, train_norm, num_nodes, device,
                ppr_graph=ppr_graph,
                ppr_edge_weight=ppr_edge_weight,
                all_node_ids=all_node_ids,
            )
            edge_metrics = evaluate_ssl_edge(
                full_embed, edge_head, valid_data_np, num_rels, eval_neg_rate,
                total_data_np, device,
                max_triples=args.eval_max_triples, rng=eval_rng,
            )

            if edge_metrics["loss"] < best_valid_loss_edge:
                best_valid_loss_edge = edge_metrics["loss"]
                best_iter = iteration
                best_ckpt = build_ssl_checkpoint(
                    model, edge_head, degree_head, iteration, args,
                    text_embeddings, ontology_embeddings, num_rels,
                    degree_bin_edges,
                    model_module="model_base4",
                    best_valid_loss_edge=best_valid_loss_edge,
                    best_iter=best_iter,
                    loss_weighter=loss_weighter,
                )
                torch.save(best_ckpt, ensure_parent(args.ssl_model_state_file))
                print(
                    f"  -> best valid_le {best_valid_loss_edge:.5f} "
                    f"@ iter {best_iter}, saved {args.ssl_model_state_file}"
                )

        optimizer.zero_grad()

    # ========================================================
    # Final: eval trên test 1 lần + lưu checkpoint cuối.
    # ========================================================
    model.eval()
    edge_head.eval()
    degree_head.eval()
    print(
        "Evaluating TEST: encode on train_graph "
        "(SSL: nhãn bậc từ train, PPR train, cạnh test không đưa vào encoder)."
    )
    full_embed = encode_full_train_graph(
        model, train_graph, train_rel, train_norm, num_nodes, device,
        ppr_graph=ppr_graph,
        ppr_edge_weight=ppr_edge_weight,
        all_node_ids=all_node_ids,
    )
    test_edge = evaluate_ssl_edge(
        full_embed, edge_head, test_data_np, num_rels, eval_neg_rate,
        total_data_np, device,
        max_triples=args.eval_max_triples, rng=eval_rng,
    )
    test_deg = evaluate_ssl_degree(full_embed, degree_head, degree_label_full)
    if loss_weighter is not None:
        w_edge, w_node = loss_weighter.weights()
    else:
        w_edge, w_node = 1.0, args.lambda_degree
    print(format_ssl_log(
        "TEST", last_iteration,
        eval_ssl_log(test_edge, test_deg, w_edge, w_node),
    ))
    if best_iter is not None:
        print(
            f"Best valid_le {best_valid_loss_edge:.5f} @ iter {best_iter} "
            f"(file: {args.ssl_model_state_file})"
        )
    else:
        print("No mid-train valid eval; best file will be the final checkpoint.")

    final_ckpt = build_ssl_checkpoint(
        model, edge_head, degree_head, last_iteration, args,
        text_embeddings, ontology_embeddings, num_rels, degree_bin_edges,
        model_module="model_base4",
        best_valid_loss_edge=(
            best_valid_loss_edge if best_iter is not None else None
        ),
        best_iter=best_iter,
        loss_weighter=loss_weighter,
    )
    final_path = ensure_parent(final_ckpt_path(args.ssl_model_state_file))
    torch.save(final_ckpt, final_path)
    print(f"SSL final checkpoint saved: {final_path}")

    # Nếu chưa từng lưu best (evaluate_every > iterations), lưu final vào path chính.
    if best_iter is None:
        torch.save(final_ckpt, ensure_parent(args.ssl_model_state_file))
        print(f"SSL checkpoint saved: {args.ssl_model_state_file}")

    print("SSL pretraining done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2 SSL upgrade (model_base4.py): degree + masked edge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data",
        dest="data",
        default="hetionet",
        help="Dataset name under fuselinker/ (e.g. hetionet) or a path",
    )
    parser.add_argument(
        "--text_embedding_file",
        dest="text_embedding_file",
        default="pubmedbert_embeddings_768.npy",
        help="Path of text embedding for each node",
    )
    parser.add_argument(
        "--knowledge_embedding_file",
        dest="knowledge_embedding_file",
        default="poincare_embeddings.npy",
        help="Path of domain knowledge embedding for each node",
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="Freeze text embedding and domain knowledge or not"
    )
    parser.add_argument(
        "--w",
        dest="w",
        type=float,
        default=0.5,
        help="The weight for fusing embedings",
    )
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
    parser.add_argument(
        "--num_hidden_layers",
        dest="num_hidden_layers",
        type=int,
        default=1,
        help="Number of hidden layers",
    )
    parser.add_argument(
        "--dropout",
        dest="dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--use_cuda",
        dest="use_cuda",
        type=bool,
        default=True,
        help="GPU",
    )
    parser.add_argument(
        "--reg_param",
        dest="reg_param",
        type=float,
        default=0.01,
        help="Regularization param (giữ để tương thích LinkPredict, không dùng trong SSL loss)",
    )
    parser.add_argument(
        "--iterations",
        dest="iterations",
        type=int,
        default=1,
        help="Number of iterations (iterations = epochs * (datasize / batchsize))",
    )
    parser.add_argument(
        "--evaluate_every",
        dest="evaluate_every",
        type=int,
        default=100,
        help="Moi N iter: encode train_graph + SSL eval tren valid. "
             "TEST cuoi cung cung encode tren train_graph (PPR train).",
    )
    parser.add_argument(
        "--eval_neg_rate",
        dest="eval_neg_rate",
        type=int,
        default=None,
        help="So negative / positive khi eval SSL edge. Mac dinh = --negative_sample.",
    )
    parser.add_argument(
        "--eval_max_triples",
        dest="eval_max_triples",
        type=int,
        default=5000,
        help="Gioi han so triple valid/test moi lan SSL edge eval.",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--graph_batch_size",
        dest="graph_batch_size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--mask_rate",
        dest="mask_rate",
        type=float,
        default=0.2,
        help="LukePi: ty le canh bi mask khoi do thi encode (default 0.2).",
    )
    parser.add_argument(
        "--negative_sample",
        dest="negative_sample",
        type=int,
        default=1,
        help="So negative / so positive (canh mask). LukePi ~1; mac dinh 1.",
    )
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

    # ======= SSL degree binning =======
    parser.add_argument(
        "--num_degree_bins",
        dest="num_degree_bins",
        type=int,
        default=10,
        help="So bucket bac (degree) cho tac vu du doan bac.",
    )
    parser.add_argument(
        "--degree_bin_strategy",
        dest="degree_bin_strategy",
        default="log",
        choices=["log", "quantile"],
        help="Cach chia bin bac: 'log' (log2, mac dinh) hoac 'quantile'.",
    )
    parser.add_argument(
        "--degree_on_full",
        dest="degree_on_full",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="Train task degree tren TOAN BO do thi train (phu deu moi node, "
             "PPR nhat quan luc train==TEST). Tat (False) neu can tiet kiem "
             "thoi gian/VRAM hoac muon giong LukePi subgraph.",
    )

    # ======= Can bang loss 2 task =======
    parser.add_argument(
        "--learn_loss_weights",
        dest="learn_loss_weights",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
        help="Tu hoc trong so 2 task theo uncertainty (Kendall). "
             "Tat (False) de dung lambda_degree tinh.",
    )
    parser.add_argument(
        "--lambda_degree",
        dest="lambda_degree",
        type=float,
        default=1.0,
        help="Trong so tinh cho loss_node khi --learn_loss_weights False "
             "(loss = loss_edge + lambda_degree * loss_node).",
    )

    parser.add_argument(
        "--ssl_model_state_file",
        dest="ssl_model_state_file",
        default=default_checkpoint(
            "hetionet", "pubmedbert", "ssl_upgrade", "ssl_model_state.pth",
        ),
        help="Checkpoint SSL upgrade (relative paths resolve under fuselinker/).",
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
        default=True,
        help="Use iterative PPR on GPU (saves RAM) instead of dense matrix inverse.",
    )
    parser.add_argument(
        "--ppr_iter_num", dest="ppr_iter_num", type=int, default=50,
        help="Number of iterations if --ppr_iterative True.",
    )
    parser.add_argument(
        "--ppr_batch_size", dest="ppr_batch_size", type=int, default=None,
        help="So walker / batch khi tinh PPR tren GPU. Mac dinh: tu chon theo VRAM.",
    )

    args = parser.parse_args()

    main(args)
