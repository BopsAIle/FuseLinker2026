"""
Đánh giá AUROC từ checkpoint đã train (không train lại).

Luồng:
  1. Load embedding cho trước (.npy) + dữ liệu test
  2. Khởi tạo LinkPredict và load trọng số từ --model_state_file
  3. Forward test graph → embedding node cuối
  4. Tính AUROC + (tùy chọn) lưu đồ thị ROC

Ví dụ:
  python eval_auroc.py --data hetionet \\
      --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \\
      --knowledge_embedding_file poincare_embeddings.npy \\
      --model_state_file checkpoints/hetionet/pubmedbert/model_state.pth \\
      --num_hidden_layers 2 --w 0.75 \\
      --roc_save_path checkpoints/hetionet/pubmedbert/roc_curve.png
"""
import task_setup  # noqa: F401
from paths import ensure_parent, resolve_data_dir, resolve_path

import argparse
import os

import numpy as np
import pandas as pd
import torch

import myutils
from calc_auroc import plot_roc_curve, roc_curve_path_from_checkpoint
from data_loader import Data


def _import_link_predict(model_module):
    if model_module == "model":
        from model import LinkPredict
        return LinkPredict, None
    if model_module == "model_base2":
        from model_base2 import LinkPredict, compute_ppr_sparse
    elif model_module == "model_base4":
        from model_base4 import LinkPredict, compute_ppr_sparse
    else:
        raise ValueError("model_module must be 'model', 'model_base2' or 'model_base4'")
    return LinkPredict, compute_ppr_sparse


def main(args):
    args.data = resolve_data_dir(args.data)
    args.model_state_file = resolve_path(args.model_state_file)
    if args.roc_save_path:
        args.roc_save_path = ensure_parent(resolve_path(args.roc_save_path))
    print(f"Data: {args.data}")
    LinkPredict, compute_ppr_sparse = _import_link_predict(args.model_module)

    train_path = f"{args.data}/train.tsv"
    valid_path = f"{args.data}/valid.tsv"
    test_path = f"{args.data}/test.tsv"
    text_embedding_path = f"{args.data}/{args.text_embedding_file}"
    knowledge_embedding_path = f"{args.data}/{args.knowledge_embedding_file}"

    train = pd.read_csv(train_path, sep="\t", header=None)
    valid = pd.read_csv(valid_path, sep="\t", header=None)
    test = pd.read_csv(test_path, sep="\t", header=None)
    graph = pd.concat([train, valid, test])

    print("Loading pretrained embedding files...")
    try:
        text_embeddings = np.load(text_embedding_path)
        print(f"Loaded text embeddings: {text_embedding_path}")
    except OSError:
        text_embeddings = None
        print("Text embeddings not found, using random init.")

    try:
        ontology_embeddings = np.load(knowledge_embedding_path)
        print(f"Loaded domain embeddings: {knowledge_embedding_path}")
    except OSError:
        ontology_embeddings = None
        print("Domain embeddings not found, using random init.")

    print("Data processing...")
    knowledge_graph = Data(graph, train, valid, test)
    num_nodes, num_rels, _ = knowledge_graph.get_stats()
    print(f"# entities: {num_nodes}, # relations: {num_rels}")

    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data
    train_data_np = knowledge_graph.train_data

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Device: {device}")

    if not os.path.isfile(args.model_state_file):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_state_file}")

    print(f"Loading checkpoint: {args.model_state_file}")
    init_kwargs = dict(
        input_dim=num_nodes,
        hidden_dim=args.n_hidden,
        num_relations=num_rels,
        num_bases=args.num_bases,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        use_cuda=args.use_cuda,
        regularization_param=args.reg_param,
        pretrained_text_embeddings=text_embeddings,
        pretrained_domain_embeddings=ontology_embeddings,
        freeze=args.freeze,
        w=args.w,
    )
    if compute_ppr_sparse is not None:
        init_kwargs["use_ppr"] = args.use_ppr
        init_kwargs["ppr_num_layers"] = args.ppr_num_layers
    model, iteration = LinkPredict.load_checkpoint(
        args.model_state_file,
        device=device,
        **init_kwargs,
    )
    if iteration is not None:
        print(f"Checkpoint iteration: {iteration}")

    # FuseLinker gốc (main.py): encode trên graph dựng từ test.tsv.
    print("Evaluating: encode on test_graph (FuseLinker original protocol).")
    eval_graph, eval_rel, eval_norm = myutils.build_graph(
        num_nodes, num_rels, test_data_np
    )
    eval_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    eval_rel = torch.from_numpy(eval_rel)
    eval_norm = myutils.node_norm_2_edge_norm(
        eval_graph, torch.from_numpy(eval_norm).view(-1, 1)
    )

    ppr_graph = None
    ppr_edge_weight = None
    all_node_ids = None
    if compute_ppr_sparse is not None and args.use_ppr:
        print("Building auxiliary PPR network (train edges)...")
        ppr_kwargs = dict(
            c=args.ppr_c, epsilon=args.ppr_eps,
            use_iterative=args.ppr_iterative,
            num_iter=args.ppr_iter_num,
        )
        try:
            ppr_graph, ppr_edge_weight = compute_ppr_sparse(
                num_nodes, train_data_np, device=device, **ppr_kwargs,
            )
        except TypeError:
            ppr_graph, ppr_edge_weight = compute_ppr_sparse(
                num_nodes, train_data_np, **ppr_kwargs,
            )
        ppr_graph = ppr_graph.to(device)
        ppr_edge_weight = ppr_edge_weight.to(device)
        all_node_ids = torch.arange(num_nodes, dtype=torch.long).view(-1, 1).to(device)

    eval_graph = eval_graph.to(device)
    eval_node_id = eval_node_id.to(device)
    eval_rel = eval_rel.to(device)
    eval_norm = eval_norm.to(device)
    test_data = torch.LongTensor(test_data_np).to(device)
    total_data = torch.LongTensor(total_data_np).to(device)

    print("Computing final embeddings and AUROC...")
    if compute_ppr_sparse is not None:
        auroc, fpr, tpr, scores, labels = model.evaluate_auroc_on_graph(
            eval_graph, eval_node_id, eval_rel, eval_norm,
            test_data, total_data,
            ppr_graph=ppr_graph,
            ppr_edge_weight=ppr_edge_weight,
            all_node_ids=all_node_ids,
            batch_size=args.eval_batch_size,
            seed=args.seed,
        )
    else:
        auroc, fpr, tpr, scores, labels = model.evaluate_auroc_on_graph(
            eval_graph, eval_node_id, eval_rel, eval_norm,
            test_data, total_data,
            batch_size=args.eval_batch_size,
            seed=args.seed,
        )

    print(f"AUROC: {auroc:.6f}")
    print(f"Samples: {len(labels)} (pos={int(labels.sum())}, neg={int((1 - labels).sum())})")

    roc_save_path = args.roc_save_path or ensure_parent(
        resolve_path(roc_curve_path_from_checkpoint(args.model_state_file))
    )
    plot_roc_curve(fpr, tpr, auroc, save_path=roc_save_path)
    print(f"ROC curve saved: {roc_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate AUROC from a trained checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="hetionet")
    parser.add_argument("--text_embedding_file", default="pubmedbert_pretrained_embeddings_768.npy")
    parser.add_argument("--knowledge_embedding_file", default="poincare_embeddings.npy")
    parser.add_argument("--model_state_file", required=True,
                        help="e.g. checkpoints/hetionet/pubmedbert/model_state.pth")
    parser.add_argument("--model_module", default="model",
                        choices=["model", "model_base2", "model_base4"])
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--w", type=float, default=0.5)
    parser.add_argument("--n_hidden", type=int, default=200)
    parser.add_argument("--num_bases", type=int, default=20)
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--reg_param", type=float, default=0.01)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roc_save_path", default=None,
                        help="ROC plot path. Default: same folder as model_state_file, file roc_curve.png")
    parser.add_argument("--use_ppr", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--ppr_c", type=float, default=0.15)
    parser.add_argument("--ppr_eps", type=float, default=1e-4)
    parser.add_argument("--ppr_num_layers", type=int, default=2)
    parser.add_argument("--ppr_iterative", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=False)
    parser.add_argument("--ppr_iter_num", type=int, default=50)

    main(parser.parse_args())
