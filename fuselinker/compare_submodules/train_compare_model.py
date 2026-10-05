"""
DistMult masked-entity training for compare_submodules/model.py.

Same sampling, loss, and test protocol as task_node_mask/train_base.py.
Checkpoints stay under compare_submodules/checkpoints/.
"""
import argparse
import importlib.util
import json
import pickle
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINTS_DIR = HERE / "checkpoints"

sys.path.insert(0, str(ROOT / "task_node_mask"))
import task_setup  # noqa: F401,E402

import numpy as np
import pandas as pd
import torch

import myutils
from calc_auroc import plot_roc_curve
from data_loader import Data
from paths import ensure_parent, resolve_data_dir


def _load_link_predict():
    path = HERE / "model.py"
    spec = importlib.util.spec_from_file_location(
        "compare_submodules_model", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LinkPredict


def _str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got {value!r}")


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _under_checkpoints(path):
    chosen = Path(path)
    if not chosen.is_absolute():
        chosen = CHECKPOINTS_DIR / chosen
    return Path(ensure_parent(str(chosen)))


def _dataset_name(data_dir):
    return Path(data_dir).name


def _embedding_name(text_embedding_file):
    return Path(text_embedding_file).stem


def _default_checkpoint(data_dir, text_embedding_file, seed, fusion_variant=None):
    parts = [
        CHECKPOINTS_DIR,
        _dataset_name(data_dir),
        _embedding_name(text_embedding_file),
        "compare_model",
    ]
    if fusion_variant:
        parts.append(fusion_variant)
    parts.extend([f"seed_{seed}", "model_state.pth"])
    return Path(*parts)


def _write_metrics(checkpoint_path, payload):
    metrics_path = Path(checkpoint_path).with_name("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Metrics saved: {metrics_path}")


def main(args):
    args.data = resolve_data_dir(args.data)
    if args.model_state_file:
        args.model_state_file = str(_under_checkpoints(args.model_state_file))
    else:
        args.model_state_file = str(ensure_parent(str(_default_checkpoint(
            args.data, args.text_embedding_file, args.seed, args.fusion_variant
        ))))
    if args.roc_save_path:
        args.roc_save_path = str(_under_checkpoints(args.roc_save_path))
    else:
        args.roc_save_path = str(
            Path(args.model_state_file).with_name("roc_curve.png")
        )

    print(f"Data: {args.data}")
    print(f"Checkpoint: {args.model_state_file}")
    print(
        "Hyperparameters | "
        f"seed={args.seed} fusion={args.fusion_variant or 'fixed'} w={args.w} n_hidden={args.n_hidden} "
        f"num_hidden_layers={args.num_hidden_layers} num_bases={args.num_bases} "
        f"dropout={args.dropout} lr={args.lr} reg_param={args.reg_param} "
        f"iterations={args.iterations} graph_batch_size={args.graph_batch_size} "
        f"graph_split_size={args.graph_split_size} "
        f"negative_sample={args.negative_sample} "
        f"eval_protocol={args.eval_protocol}"
    )

    train_path = f"{args.data}/train.tsv"
    valid_path = f"{args.data}/valid.tsv"
    test_path = f"{args.data}/test.tsv"
    text_embedding_path = f"{args.data}/{args.text_embedding_file}"
    knowledge_embedding_path = f"{args.data}/{args.knowledge_embedding_file}"

    train = pd.read_csv(train_path, sep="\t", header=None)
    valid = pd.read_csv(valid_path, sep="\t", header=None)
    test = pd.read_csv(test_path, sep="\t", header=None)
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
    print("# entities:", num_nodes)
    print("# relations:", num_rels)
    print("# edges:", num_edges)
    with open(f"{args.data}/relation2index.pkl", "wb") as file:
        pickle.dump(knowledge_graph.relation2index, file)
    with open(f"{args.data}/index2relation.pkl", "wb") as file:
        pickle.dump(knowledge_graph.index2relation, file)
    with open(f"{args.data}/entity2index.pkl", "wb") as file:
        pickle.dump(knowledge_graph.entity2index, file)
    with open(f"{args.data}/index2entity.pkl", "wb") as file:
        pickle.dump(knowledge_graph.index2entity, file)

    train_data_np = knowledge_graph.train_data
    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data

    _set_seed(args.seed)
    LinkPredict = _load_link_predict()
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
        freeze=args.freeze,
        w=args.w,
        fusion_variant=args.fusion_variant,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    print(device)
    print("Task 1 DistMult | Encoder: compare_submodules/model.py")

    train_graph, train_rel, train_norm = myutils.build_graph(
        num_nodes, num_rels, train_data_np
    )
    train_deg = train_graph.in_degrees(
        range(train_graph.number_of_nodes())
    ).float().view(-1, 1)

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
        edge_norm = myutils.node_norm_2_edge_norm(
            g, torch.from_numpy(node_norm).view(-1, 1)
        )
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
        {"state_dict": model.state_dict(), "iteration": iteration},
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

    with torch.no_grad():
        output = model(eval_graph, eval_node_id, eval_rel, eval_norm)

    old_time = time.time()
    hits = [1, 3, 10]
    mr, mrr, hits_dict = myutils.calc_mrr(
        output, model.relation_weights, test_data,
        total_data,
        batch_size=args.eval_batch_size,
        neg_sample_size_eval=args.neg_sample_size_eval,
        hits=hits, eval_p=args.eval_protocol,
    )
    print(time.time() - old_time)

    print(f"MR: {mr:.6f}")
    print(f"MRR: {mrr:.6f}")
    for key, value in hits_dict.items():
        print(f"Hits @ {key} = {value:.6f}")

    auroc, fpr, tpr, _, _ = model.evaluate_auroc(
        output, test_data, total_data, seed=args.seed,
    )
    print(f"AUROC: {auroc:.6f}")

    plot_roc_curve(fpr, tpr, auroc, save_path=ensure_parent(args.roc_save_path))
    print(f"ROC curve saved: {args.roc_save_path}")

    _write_metrics(args.model_state_file, {
        "encoder": "model",
        "seed": args.seed,
        "fusion_variant": args.fusion_variant or "fixed",
        "mr": float(mr),
        "mrr": float(mrr),
        "hits": {str(key): float(value) for key, value in hits_dict.items()},
        "auroc": float(auroc),
    })
    print("Training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline: DistMult masked-entity prediction (compare_submodules/model.py)",
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
        "--fusion_variant", dest="fusion_variant", default=None,
        choices=["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"],
        help="Replace fixed w-mix with a fusion class from 10_fusion_variants.md",
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
        "--use_cuda", dest="use_cuda", type=_str2bool, default=True, help="GPU",
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
        "--neg_sample_size_eval", dest="neg_sample_size_eval", type=int, default=100,
    )
    parser.add_argument(
        "--eval_protocol", dest="eval_protocol", default="filtered",
    )
    parser.add_argument(
        "--seed", dest="seed", type=int, default=42,
        help="Seed for model init and negative sampling",
    )
    parser.add_argument(
        "--model_state_file", dest="model_state_file", default=None,
        help="Checkpoint path. Relative paths resolve under compare_submodules/checkpoints/.",
    )
    parser.add_argument(
        "--roc_save_path", dest="roc_save_path", default=None,
        help="ROC plot path. Default: roc_curve.png beside the checkpoint",
    )
    main(parser.parse_args())
