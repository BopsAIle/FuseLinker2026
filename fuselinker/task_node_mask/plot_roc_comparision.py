"""
plot_roc_comparison.py
======================

V� � th� ROC so s�nh baseline (FuseLinker) v� upgrade (Multimodal FuseLinker
with Auxiliary PPR Diffusion) tr�n nhi�u t�p d� li�u + nhi�u ngu�n embedding.

C�u tr�c th� m�c checkpoint (m�c �nh):

    <root>/
        <dataset>/                              # v� d�: hetionet, suppkg, kegg50k
            train.tsv, valid.tsv, test.tsv
            poincare_embeddings.npy             # domain knowledge
            <text_emb>.npy                      # text embedding
            checkpoints/
                <embedding>/                    # pubmedbert, flant5, llama2, ...
                    base/
                        model_state.pth
                    upgrade/
                        model_state.pth
                ...

�u ra:
    - M�t figure g�m ROWS x COLS subplot (ROWS = s� embedding, COLS = s� dataset).
    - M�i subplot: 2 ��ng ROC (Base / Upgrade) + AUROC trong legend.
    - L�u ra file PNG (m�c �nh: roc_comparison.png).

S� d�ng:

    python plot_roc_comparison.py \\
        --root . \\
        --datasets hetionet suppkg kegg50k \\
        --embeddings pubmedbert flant5 llama2 pmcllama bert \\
        --output roc_comparison.png

N�u m�t t� h�p (dataset, embedding) kh�ng c� checkpoint (base ho�c upgrade),
subplot t��ng �ng s� ��c v� tr�ng v� in c�nh b�o. B�n c� th� cung c�p
--skip-missing � b� qua h�n nh�ng t� h�p � (thay v� � � tr�ng).
"""

from __future__ import annotations

import task_setup  # noqa: F401
from paths import ROOT, ensure_parent, resolve_path

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Kh�ng c�n GUI khi l�u file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import myutils
from calc_auroc import compute_roc_curve
from checkpoint_utils import (
    infer_checkpoint_specs,
    load_checkpoint_bundle,
    validate_eval_against_checkpoint,
)
from data_loader import Data


# ---------------------------------------------------------------------------
# C�u h�nh m�c �nh: �nh x� t�n embedding -> t�n file .npy + dim
# (B�n c� th� override qua --embedding-file / --embedding-dim n�u kh�c)
# ---------------------------------------------------------------------------
EMBEDDING_DEFAULTS: Dict[str, Dict[str, object]] = {
    "pubmedbert": {"file": "pubmedbert_pretrained_embeddings_768.npy", "dim": 768},
    "flant5":     {"file": "flant5_pretrained_embeddings_768.npy",     "dim": 768},
    "bert":       {"file": "bert_pretrained_embeddings_768.npy",       "dim": 768},
    "llama2":     {"file": "llama2_pretrained_embeddings_4096.npy",    "dim": 4096},
    "pmcllama":   {"file": "pmcllama_pretrained_embeddings_4096.npy",  "dim": 4096},
    "medllama":   {"file": "medllama_pretrained_embeddings_4096.npy",  "dim": 4096},
}

DEFAULT_DOMAIN_FILE = "poincare_embeddings.npy"


# ---------------------------------------------------------------------------
# C�u tr�c d� li�u
# ---------------------------------------------------------------------------
@dataclass
class CheckpointResult:
    """K�t qu� AUROC + ROC c�a 1 checkpoint."""
    fpr: np.ndarray
    tpr: np.ndarray
    auroc: float
    scores: np.ndarray
    labels: np.ndarray
    iteration: Optional[int]


@dataclass
class DatasetConfig:
    """Th�ng tin � load 1 dataset."""
    name: str
    path: str


# ---------------------------------------------------------------------------
# Ti�n �ch
# ---------------------------------------------------------------------------
def _import_link_predict(model_module: str):
    """Import LinkPredict v� compute_ppr_sparse t��ng �ng v�i model_module."""
    if model_module == "model":
        from model import LinkPredict
        return LinkPredict, None
    if model_module == "model_base2":
        from model_base2 import LinkPredict, compute_ppr_sparse
    elif model_module == "model_base4":
        from model_base4 import LinkPredict, compute_ppr_sparse
    else:
        raise ValueError(
            f"model_module kh�ng h�p l�: {model_module}. "
            f"Ph�i l� 'model', 'model_base2' ho�c 'model_base4'."
        )
    return LinkPredict, compute_ppr_sparse


def _detect_model_module(ckpt_path: str) -> str:
    """Nh�n di�n model module t� state_dict c�a checkpoint."""
    bundle = load_checkpoint_bundle(ckpt_path, map_location="cpu")
    return infer_checkpoint_specs(bundle).get("model_module", "model_base4")


def _resolve_embedding_file(embedding_name: str, dataset_path: str,
                           override_file: Optional[str]) -> str:
    """Tr� v� ��ng d�n tuy�t �i c�a file embedding."""
    if override_file:
        fname = override_file
    elif embedding_name in EMBEDDING_DEFAULTS:
        fname = EMBEDDING_DEFAULTS[embedding_name]["file"]
    else:
        raise FileNotFoundError(
            f"Kh�ng bi�t file embedding cho '{embedding_name}'. "
            f"Th�m v�o EMBEDDING_DEFAULTS ho�c d�ng --embedding-file."
        )
    return os.path.join(dataset_path, fname)


def _load_optional(path: str):
    """Load numpy file, tr� v� None n�u kh�ng t�n t�i."""
    if not os.path.isfile(path):
        return None
    return np.load(path)


# ---------------------------------------------------------------------------
# �nh gi� 1 checkpoint
# ---------------------------------------------------------------------------
def evaluate_checkpoint(
    ckpt_path: str,
    dataset_path: str,
    embedding_name: str,
    embedding_file_override: Optional[str],
    n_hidden: int,
    num_bases: int,
    num_hidden_layers: int,
    dropout: float,
    reg_param: float,
    eval_batch_size: int,
    seed: int,
    ppr_c: float,
    ppr_eps: float,
    ppr_num_layers: int,
    use_cuda: bool,
) -> Tuple[CheckpointResult, dict]:
    """
    Load checkpoint, forward test graph, t�nh AUROC + ROC.

    Returns
    -------
    CheckpointResult : K�t qu� ROC + AUROC.
    info             : dict th�ng tin th�m (model_module, dataset stats, ...).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint kh�ng t�n t�i: {ckpt_path}")

    model_module = _detect_model_module(ckpt_path)
    LinkPredict, compute_ppr_sparse = _import_link_predict(model_module)

    # ----- Load triples -----
    train = pd.read_csv(os.path.join(dataset_path, "train.tsv"),
                        sep="\t", header=None)
    valid = pd.read_csv(os.path.join(dataset_path, "valid.tsv"),
                        sep="\t", header=None)
    test = pd.read_csv(os.path.join(dataset_path, "test.tsv"),
                       sep="\t", header=None)
    graph = pd.concat([train, valid, test])

    # ----- Load embeddings -----
    text_path = _resolve_embedding_file(embedding_name, dataset_path,
                                        embedding_file_override)
    domain_path = os.path.join(dataset_path, DEFAULT_DOMAIN_FILE)
    text_embeddings = _load_optional(text_path)
    ontology_embeddings = _load_optional(domain_path)
    if text_embeddings is None:
        raise FileNotFoundError(
            f"File embedding vn b�n kh�ng t�n t�i: {text_path}"
        )
    if ontology_embeddings is None:
        raise FileNotFoundError(
            f"File domain embedding kh�ng t�n t�i: {domain_path}"
        )

    # ----- Build Data object -----
    knowledge_graph = Data(graph, train, valid, test)
    num_nodes, num_rels, _ = knowledge_graph.get_stats()

    train_data_np = knowledge_graph.train_data
    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data

    # ----- Device -----
    use_cuda_eff = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda_eff else "cpu")

    # ----- Validate -----
    ckpt_specs = infer_checkpoint_specs(load_checkpoint_bundle(ckpt_path))
    embedding_file_basename = os.path.basename(text_path)
    validate_eval_against_checkpoint(
        ckpt_specs, text_embeddings, ontology_embeddings,
        embedding_file_basename, DEFAULT_DOMAIN_FILE, n_hidden,
    )

    # ----- Build + load model -----
    init_kwargs = dict(
        input_dim=num_nodes,
        hidden_dim=n_hidden,
        num_relations=num_rels,
        num_bases=num_bases,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
        use_cuda=use_cuda_eff,
        regularization_param=reg_param,
        pretrained_text_embeddings=text_embeddings,
        pretrained_domain_embeddings=ontology_embeddings,
        freeze=False,
        w=0.5,
    )
    if compute_ppr_sparse is not None:
        init_kwargs["use_ppr"] = True
        init_kwargs["ppr_num_layers"] = ppr_num_layers

    try:
        model, iteration = LinkPredict.load_checkpoint(
            ckpt_path, device=device, **init_kwargs,
        )
    except Exception as exc:
        # Retry with relaxed state_dict loading (non-strict) when architecture
        # in checkpoint differs slightly from current code. This may produce
        # a best-effort model compatible with evaluation.
        print(f"  [WARN] load_checkpoint failed: {type(exc).__name__}: {exc}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = LinkPredict(**init_kwargs)
        state = checkpoint.get("state_dict", checkpoint)
        try:
            model.load_state_dict(state, strict=False)
            model.to(device)
            iteration = checkpoint.get("iteration", None)
            print("  [INFO] Loaded checkpoint with strict=False (best-effort).")
        except Exception as exc2:
            raise RuntimeError(f"Failed to load checkpoint even with strict=False: {exc2}")

    # FuseLinker gốc (main.py): encode trên graph dựng từ test.tsv.
    eval_graph, eval_rel_np, eval_norm_np = myutils.build_graph(
        num_nodes, num_rels, test_data_np
    )
    eval_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    eval_rel = torch.from_numpy(eval_rel_np).to(device)
    eval_norm = myutils.node_norm_2_edge_norm(
        eval_graph, torch.from_numpy(eval_norm_np).view(-1, 1)
    ).to(device)
    eval_graph = eval_graph.to(device)
    eval_node_id = eval_node_id.to(device)

    # ----- Build PPR auxiliary graph (n�u c�n) -----
    ppr_graph = None
    ppr_edge_weight = None
    all_node_ids = None
    if compute_ppr_sparse is not None:
        ppr_graph, ppr_edge_weight = compute_ppr_sparse(
            num_nodes, train_data_np,
            c=ppr_c, epsilon=ppr_eps,
            use_iterative=False, num_iter=50,
        )
        ppr_graph = ppr_graph.to(device)
        ppr_edge_weight = ppr_edge_weight.to(device)
        all_node_ids = torch.arange(num_nodes, dtype=torch.long).view(-1, 1).to(device)

    test_data = torch.LongTensor(test_data_np).to(device)
    total_data = torch.LongTensor(total_data_np).to(device)

    # ----- Forward + AUROC -----
    if compute_ppr_sparse is not None:
        auroc, fpr, tpr, scores, labels = model.evaluate_auroc_on_graph(
            eval_graph, eval_node_id, eval_rel, eval_norm,
            test_data, total_data,
            ppr_graph=ppr_graph,
            ppr_edge_weight=ppr_edge_weight,
            all_node_ids=all_node_ids,
            batch_size=eval_batch_size,
            seed=seed,
        )
    else:
        auroc, fpr, tpr, scores, labels = model.evaluate_auroc_on_graph(
            eval_graph, eval_node_id, eval_rel, eval_norm,
            test_data, total_data,
            batch_size=eval_batch_size,
            seed=seed,
        )

    info = {
        "model_module": model_module,
        "num_nodes": num_nodes,
        "num_rels": num_rels,
        "iteration": iteration,
        "ckpt_path": ckpt_path,
        "embedding_file": embedding_file_basename,
        "n_pos": int(np.sum(labels == 1)),
        "n_neg": int(np.sum(labels == 0)),
    }
    return CheckpointResult(fpr=fpr, tpr=tpr, auroc=float(auroc),
                            scores=scores, labels=labels,
                            iteration=iteration), info


# ---------------------------------------------------------------------------
# V� grid ROC
# ---------------------------------------------------------------------------
def plot_roc_grid(
    results_grid: Dict[Tuple[str, str], Dict[str, CheckpointResult]],
    datasets: List[str],
    embeddings: List[str],
    output_path: str,
    title: Optional[str] = None,
    dpi: int = 150,
    figsize_per_cell: Tuple[float, float] = (3.5, 3.5),
) -> None:
    """
    V� figure ROWS x COLS:
        - ROWS = embeddings
        - COLS = datasets
    M�i subplot so s�nh Base vs Upgrade.
    """
    n_rows = len(embeddings)
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_rows),
        squeeze=False,
    )

    for i, emb in enumerate(embeddings):
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            cell = results_grid.get((ds, emb), {})
            base_res = cell.get("base")
            up_res = cell.get("upgrade")

            if base_res is not None:
                ax.plot(base_res.fpr, base_res.tpr,
                        color="#1f77b4", lw=1.8,
                        label=f"Base (AUROC={base_res.auroc:.4f})")
            if up_res is not None:
                ax.plot(up_res.fpr, up_res.tpr,
                        color="#d62728", lw=1.8, linestyle="--",
                        label=f"Upgrade (AUROC={up_res.auroc:.4f})")

            ax.plot([0, 1], [0, 1], color="gray", lw=0.8,
                    linestyle=":", alpha=0.7)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.02])
            ax.set_aspect("equal", adjustable="box")

            # Ch� show x/y labels � bi�n ngo�i
            if i == n_rows - 1:
                ax.set_xlabel("False Positive Rate")
            if j == 0:
                ax.set_ylabel("True Positive Rate")

            # Title cho c�t tr�n c�ng: t�n dataset
            # Title cho h�ng tr�i nh�t: t�n embedding
            if i == 0:
                ax.set_title(ds, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(
                    f"{emb}\n\nTrue Positive Rate",
                    fontsize=9,
                )

            if base_res is not None or up_res is not None:
                ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
                ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
            else:
                ax.text(0.5, 0.5, "No checkpoint",
                        ha="center", va="center", transform=ax.transAxes,
                        color="gray", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.supxlabel("False Positive Rate", fontsize=11)
    fig.supylabel("Embedding (row) | True Positive Rate (col 0)",
                  fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] � l�u figure: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="V� � th� ROC so s�nh Base vs Upgrade tr�n nhi�u dataset + embedding.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default=str(ROOT),
                   help="Thu muc goc chua cac dataset (mac dinh: fuselinker/).")
    p.add_argument("--datasets", nargs="+",
                   default=["hetionet", "suppkg", "kegg50k"],
                   help="Danh sach ten dataset (thu muc con trong root).")
    p.add_argument("--embeddings", nargs="+",
                   default=["pubmedbert", "flant5", "llama2", "pmcllama", "bert"],
                   help="Danh sach embedding.")
    p.add_argument("--checkpoint-dirname", default="checkpoints",
                   help="Ten thu muc checkpoint (fuselinker/checkpoints/...).")
    p.add_argument("--checkpoint-filename", default="model_state.pth",
                   help="Ten file checkpoint.")
    p.add_argument("--embedding-file", default=None,
                   help="Override file embedding. Neu khong, dung EMBEDDING_DEFAULTS.")
    p.add_argument("--variants", nargs="+", default=["model", "modelbase"],
                   help="Cac bien the checkpoint (subfolder: model=base, modelbase=upgrade).")
    p.add_argument("--output", default="roc_comparison.png",
                   help="File PNG dau ra (relative -> under fuselinker/).")
    p.add_argument("--title", default="ROC Comparison: Base vs Upgrade",
                   help="Tieu de cua figure.")

    # Si�u tham s� m� h�nh (ph�i kh�p l�c train)
    p.add_argument("--n-hidden", type=int, default=200)
    p.add_argument("--num-bases", type=int, default=20)
    p.add_argument("--num-hidden-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--reg-param", type=float, default=0.01)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)

    # PPR
    p.add_argument("--ppr-c", type=float, default=0.15)
    p.add_argument("--ppr-eps", type=float, default=1e-4)
    p.add_argument("--ppr-num-layers", type=int, default=2)

    # Hardware
    p.add_argument("--use-cuda", action="store_true",
                   help="D�ng GPU n�u c� s�n.")
    p.add_argument("--skip-missing", action="store_true",
                   help="B� qua (dataset, embedding) kh�ng c� � checkpoint "
                        "thay v� v� � tr�ng.")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    args.root = resolve_path(args.root)
    args.output = ensure_parent(resolve_path(args.output))
    print("=" * 70)
    print(f"Root:      {args.root}")
    print(f"Datasets:  {args.datasets}")
    print(f"Embeddings: {args.embeddings}")
    print(f"Variants:  {args.variants}")
    print("=" * 70)

    results_grid: Dict[Tuple[str, str], Dict[str, CheckpointResult]] = {}

    summary_rows: List[dict] = []

    for ds in args.datasets:
        dataset_path = os.path.join(args.root, ds)
        for emb in args.embeddings:
            cell_results: Dict[str, CheckpointResult] = {}
            for variant in args.variants:
                primary = os.path.join(
                    dataset_path, args.checkpoint_dirname, emb, variant,
                    args.checkpoint_filename,
                )
                # Try a couple of common alternative layouts: root/checkpoints/<dataset>/... and root/checkpoints/<embedding>/...
                alt1 = os.path.join(
                    args.root, args.checkpoint_dirname, ds, emb, variant, args.checkpoint_filename
                )
                alt2 = os.path.join(
                    args.root, args.checkpoint_dirname, emb, variant, args.checkpoint_filename
                )

                ckpt_path = None
                for p in (primary, alt1, alt2):
                    if os.path.isfile(p):
                        ckpt_path = p
                        break
                if ckpt_path is None:
                    print(f"[SKIP] Kh�ng th�y checkpoint: {primary}  (tried alt: {alt1}, {alt2})")
                    continue
                print(f"\n[EVAL] dataset={ds} | embedding={emb} | variant={variant}")
                try:
                    res, info = evaluate_checkpoint(
                        ckpt_path=ckpt_path,
                        dataset_path=dataset_path,
                        embedding_name=emb,
                        embedding_file_override=args.embedding_file,
                        n_hidden=args.n_hidden,
                        num_bases=args.num_bases,
                        num_hidden_layers=args.num_hidden_layers,
                        dropout=args.dropout,
                        reg_param=args.reg_param,
                        eval_batch_size=args.eval_batch_size,
                        seed=args.seed,
                        ppr_c=args.ppr_c,
                        ppr_eps=args.ppr_eps,
                        ppr_num_layers=args.ppr_num_layers,
                        use_cuda=args.use_cuda,
                    )
                except Exception as exc:
                    print(f"  [ERROR] {type(exc).__name__}: {exc}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
                    continue
                cell_results[variant] = res
                summary_rows.append({
                    "dataset": ds,
                    "embedding": emb,
                    "variant": variant,
                    "auroc": res.auroc,
                    "n_pos": info["n_pos"],
                    "n_neg": info["n_neg"],
                    "model_module": info["model_module"],
                    "iteration": info["iteration"],
                    "ckpt_path": ckpt_path,
                })
                print(f"  AUROC = {res.auroc:.4f}  "
                      f"(pos={info['n_pos']}, neg={info['n_neg']}, "
                      f"module={info['model_module']})")
            if cell_results:
                results_grid[(ds, emb)] = cell_results

    if not results_grid:
        print("[ERROR] Kh�ng c� checkpoint n�o ��c load th�nh c�ng.")
        sys.exit(1)

    # N�u skip-missing: l�c b� nh�ng � kh�ng c� � c� base+upgrade
    if args.skip_missing:
        filtered = {}
        for key, val in results_grid.items():
            if all(v in val for v in args.variants):
                filtered[key] = val
            else:
                print(f"[SKIP-EMPTY] B� � kh�ng � bi�n th�: {key}")
        results_grid = filtered

    # V�
    plot_roc_grid(
        results_grid=results_grid,
        datasets=args.datasets,
        embeddings=args.embeddings,
        output_path=args.output,
        title=args.title,
    )

    # L�u summary ra CSV
    if summary_rows:
        import csv
        summary_csv = os.path.splitext(args.output)[0] + "_summary.csv"
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[OK] � l�u summary: {summary_csv}")


if __name__ == "__main__":
    main(parse_args())
