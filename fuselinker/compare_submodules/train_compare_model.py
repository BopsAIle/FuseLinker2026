"""
DistMult masked-entity training for compare_submodules/model.py.

Same sampling, loss, and test protocol as task_node_mask/train_base.py.
Checkpoints stay under compare_submodules/checkpoints/.
"""
import argparse
import hashlib
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

import dgl
import numpy as np
import pandas as pd
import torch

import myutils
from calc_auroc import plot_roc_curve
from data_loader import Data
from paths import ensure_parent, resolve_data_dir


def _load_model_module():
    path = HERE / "model.py"
    spec = importlib.util.spec_from_file_location(
        "compare_submodules_model", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _default_checkpoint(
    data_dir, text_embedding_file, seed, fusion_variant=None,
    use_domain_mlp_projector=False, use_residual_rgcn=False, use_ppr=False,
):
    parts = [
        CHECKPOINTS_DIR,
        _dataset_name(data_dir),
        _embedding_name(text_embedding_file),
        "compare_model",
    ]
    if use_domain_mlp_projector:
        parts.append("domain_mlp_projector")
    if use_residual_rgcn:
        parts.append("residual_rgcn")
    if use_ppr:
        parts.append("ppr")
    if fusion_variant:
        parts.append(fusion_variant)
    parts.extend([f"seed_{seed}", "model_state.pth"])
    return Path(*parts)


def _ppr_cache_path(data_dir, train_data_np, args):
    digest = hashlib.md5(np.ascontiguousarray(train_data_np).tobytes()).hexdigest()[:12]
    name = (
        f"topk{args.ppr_topk}_c{args.ppr_c}_iter{args.ppr_num_iter}"
        f"_eps{args.ppr_eps}_batch{args.ppr_batch_size}_{digest}.pt"
    )
    return CHECKPOINTS_DIR / _dataset_name(data_dir) / "ppr_cache" / name


def _load_or_build_ppr(model_module, args, num_nodes, train_data_np, device):
    cache_path = _ppr_cache_path(args.data, train_data_np, args)
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu")
        print(f"Loaded PPR cache: {cache_path}")
        graph = dgl.graph(
            (payload["src"], payload["dst"]),
            num_nodes=int(payload["num_nodes"]),
        )
        return graph, payload["weight"]

    print(
        f"Building CUDA PPR top-{args.ppr_topk} on {device} "
        f"(c={args.ppr_c}, epsilon={args.ppr_eps}, "
        f"num_iter={args.ppr_num_iter}, batch_size={args.ppr_batch_size})..."
    )
    graph, weight = model_module.compute_ppr_sparse(
        num_nodes,
        train_data_np,
        c=args.ppr_c,
        epsilon=args.ppr_eps,
        num_iter=args.ppr_num_iter,
        topk=args.ppr_topk,
        device=device,
        batch_size=args.ppr_batch_size,
    )
    print(
        f"Gppr: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, device={graph.device}."
    )
    src, dst = graph.edges()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "src": src.detach().cpu(),
            "dst": dst.detach().cpu(),
            "weight": weight.detach().cpu(),
            "num_nodes": int(num_nodes),
        },
        cache_path,
    )
    print(f"Saved PPR cache: {cache_path}")
    graph = graph.cpu()
    weight = weight.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return graph, weight


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
            args.data, args.text_embedding_file, args.seed, args.fusion_variant,
            args.use_domain_mlp_projector, args.use_residual_rgcn, args.use_ppr,
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
        f"seed={args.seed} fusion={args.fusion_variant or 'fixed'} "
        f"domain_mlp_projector={args.use_domain_mlp_projector} "
        f"residual_rgcn={args.use_residual_rgcn} "
        f"ppr={args.use_ppr} ppr_num_layers={args.ppr_num_layers} "
        f"ppr_c={args.ppr_c} ppr_eps={args.ppr_eps} "
        f"ppr_num_iter={args.ppr_num_iter} ppr_topk={args.ppr_topk} "
        f"ppr_batch_size={args.ppr_batch_size} "
        f"w={args.w} n_hidden={args.n_hidden} "
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_module = _load_model_module()
    ppr_graph = None
    ppr_edge_weight = None
    all_node_ids = None
    if args.use_ppr:
        if device.type != "cuda":
            raise RuntimeError("PPR is built on CUDA, but CUDA is not available.")
        ppr_graph, ppr_edge_weight = _load_or_build_ppr(
            model_module, args, num_nodes, train_data_np, device,
        )
        ppr_graph = ppr_graph.to(device)
        ppr_edge_weight = ppr_edge_weight.to(device)
        all_node_ids = torch.arange(num_nodes, dtype=torch.long).view(-1, 1).to(device)

    _set_seed(args.seed)
    model = model_module.LinkPredict(
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
        use_domain_mlp_projector=args.use_domain_mlp_projector,
        use_residual_rgcn=args.use_residual_rgcn,
        use_ppr=args.use_ppr,
        ppr_num_layers=args.ppr_num_layers,
    )

    model = model.to(device)
    ppr_kwargs = dict(
        ppr_graph=ppr_graph,
        ppr_edge_weight=ppr_edge_weight,
        all_node_ids=all_node_ids,
    )
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

    eval_only = args.eval_only and Path(args.model_state_file).is_file()
    if eval_only:
        checkpoint = torch.load(args.model_state_file, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint for eval: {args.model_state_file}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Start training (masked-entity DistMult)...")

    if not eval_only:
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

            embed = model(g, node_id, edge_type, edge_norm, **ppr_kwargs)
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
        del optimizer, g, node_id, edge_type, edge_norm, data, labels, embed, loss

    print(
        "Evaluating TEST: encode on test_graph "
        "(FuseLinker original main.py protocol)."
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
        output = model(eval_graph, eval_node_id, eval_rel, eval_norm, **ppr_kwargs)

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
        "use_domain_mlp_projector": args.use_domain_mlp_projector,
        "domain_map": (
            "domain_mlp_projector" if args.use_domain_mlp_projector else "linear"
        ),
        "use_residual_rgcn": args.use_residual_rgcn,
        "rgcn_block": (
            "residual_rgcn" if args.use_residual_rgcn else "rel_graph_conv"
        ),
        "use_ppr": args.use_ppr,
        "ppr_num_layers": args.ppr_num_layers,
        "ppr_c": args.ppr_c,
        "ppr_eps": args.ppr_eps,
        "ppr_num_iter": args.ppr_num_iter,
        "ppr_topk": args.ppr_topk,
        "ppr_batch_size": args.ppr_batch_size,
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
        "--use_domain_mlp_projector", dest="use_domain_mlp_projector",
        type=_str2bool, default=False,
        help="Replace the linear domain map with Domain MLP Projector",
    )
    parser.add_argument(
        "--use_residual_rgcn", dest="use_residual_rgcn",
        type=_str2bool, default=False,
        help="Replace RelGraphConv hidden layers with Residual R-GCN blocks",
    )
    parser.add_argument(
        "--use_ppr", dest="use_ppr", type=_str2bool, default=False,
        help="Fuse a Personalized PageRank branch with the R-GCN output",
    )
    parser.add_argument(
        "--ppr_num_layers", dest="ppr_num_layers", type=int, default=2,
    )
    parser.add_argument("--ppr_c", dest="ppr_c", type=float, default=0.15)
    parser.add_argument("--ppr_eps", dest="ppr_eps", type=float, default=1e-4)
    parser.add_argument("--ppr_num_iter", dest="ppr_num_iter", type=int, default=50)
    parser.add_argument("--ppr_topk", dest="ppr_topk", type=int, default=50)
    parser.add_argument(
        "--ppr_batch_size", dest="ppr_batch_size", type=int, default=128,
    )
    parser.add_argument(
        "--eval_only", dest="eval_only", type=_str2bool, default=False,
        help="Load an existing checkpoint and run test evaluation",
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
