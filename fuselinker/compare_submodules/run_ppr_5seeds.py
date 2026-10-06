"""Compare model.py vs model.py + PPR on suppKG.

Five seeds (42-46) and two text embeddings:
bert_pretrained_embeddings_768 and pubmedbert_pretrained_embeddings_768.

Checkpoints:
  compare_submodules/checkpoints/suppkg/<embedding>/compare_model/seed_<seed>/
  compare_submodules/checkpoints/suppkg/<embedding>/compare_model/ppr/seed_<seed>/

PPR graph cache, shared by both embeddings:
  compare_submodules/checkpoints/suppkg/ppr_cache/

Notes:
  compare_submodules/ppr_5seed_suppkg/
"""
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
OUT_DIR = HERE / "ppr_5seed_suppkg"
LOG_DIR = OUT_DIR / "logs"
CHECKPOINT_ROOT = HERE / "checkpoints" / "suppkg"

SEEDS = [42, 43, 44, 45, 46]
EMBEDDINGS = [
    "bert_pretrained_embeddings_768",
    "pubmedbert_pretrained_embeddings_768",
]
VARIANTS = [
    {
        "name": "model",
        "label": "model.py",
        "flag": False,
        "checkpoint": "compare_model/seed_{seed}/metrics.json",
    },
    {
        "name": "model_plus_ppr",
        "label": "model.py + PPR",
        "flag": True,
        "checkpoint": "compare_model/ppr/seed_{seed}/metrics.json",
    },
]
METRIC_FIELDS = ["mr", "mrr", "hits_1", "hits_3", "hits_10", "auroc"]
PROTOCOL = (
    "suppKG (data=suppkg), 40000 iterations, evaluate_every=1000, "
    "num_hidden_layers=2, w=0.75, knowledge=poincare_embeddings.npy, "
    "fusion=fixed, domain map=Linear, hidden layers=RelGraphConv. "
    "PPR is a CUDA top-50 Personalized PageRank branch "
    "(c=0.15, epsilon=1e-4, 50 iterations, batch_size=128, 2 PPR layers) "
    "fused with the R-GCN output by a zero-init gate starting at 0.75/0.25 "
    "when --use_ppr true."
)


def _flatten(payload):
    hits = payload["hits"]
    return {
        "mr": float(payload["mr"]),
        "mrr": float(payload["mrr"]),
        "hits_1": float(hits["1"]),
        "hits_3": float(hits["3"]),
        "hits_10": float(hits["10"]),
        "auroc": float(payload["auroc"]),
    }


def _mean(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in METRIC_FIELDS}


def _format(metrics):
    return (
        f"MR {metrics['mr']:.6f} | MRR {metrics['mrr']:.6f} | "
        f"Hits@1 {metrics['hits_1']:.6f} | Hits@3 {metrics['hits_3']:.6f} | "
        f"Hits@10 {metrics['hits_10']:.6f} | AUROC {metrics['auroc']:.6f}"
    )


def _metrics_path(embedding, variant, seed):
    relative = variant["checkpoint"].format(seed=seed)
    return CHECKPOINT_ROOT / embedding / relative


def _train(embedding, variant, seed):
    metrics_path = _metrics_path(embedding, variant, seed)
    if metrics_path.is_file():
        print(f"SKIP {variant['name']} {embedding} seed={seed} ({metrics_path})", flush=True)
        return
    checkpoint = metrics_path.with_name("model_state.pth")
    command = [
        str(PYTHON), "-u", str(HERE / "train_compare_model.py"),
        "--data", "suppkg",
        "--seed", str(seed),
        "--iterations", "40000",
        "--evaluate_every", "1000",
        "--num_hidden_layers", "2",
        "--w", "0.75",
        "--text_embedding_file", f"{embedding}.npy",
        "--knowledge_embedding_file", "poincare_embeddings.npy",
        "--use_domain_mlp_projector", "false",
        "--use_residual_rgcn", "false",
        "--use_ppr", "true" if variant["flag"] else "false",
        "--eval_only", "true" if checkpoint.is_file() else "false",
    ]
    log_path = LOG_DIR / f"{variant['name']}__{embedding}__seed_{seed}.log"
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    print(f"START {variant['name']} {embedding} seed={seed}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant['name']} {embedding} seed={seed} failed with code "
            f"{completed.returncode}. See {log_path}"
        )
    print(f"DONE {variant['name']} {embedding} seed={seed}", flush=True)


def _load():
    report = {}
    for embedding in EMBEDDINGS:
        report[embedding] = {}
        for variant in VARIANTS:
            runs = []
            missing = []
            for seed in SEEDS:
                path = _metrics_path(embedding, variant, seed)
                if not path.is_file():
                    missing.append(seed)
                    continue
                with path.open(encoding="utf-8") as handle:
                    runs.append({
                        "seed": seed,
                        "metrics": _flatten(json.load(handle)),
                        "path": str(path),
                    })
            report[embedding][variant["name"]] = {
                "label": variant["label"],
                "runs": runs,
                "missing_seeds": missing,
            }
            if len(runs) == len(SEEDS):
                report[embedding][variant["name"]]["mean"] = _mean(
                    [item["metrics"] for item in runs]
                )
    return report


def _table_row(name, metrics):
    return (
        f"| {name} | {metrics['mr']:.6f} | {metrics['mrr']:.6f} | "
        f"{metrics['hits_1']:.6f} | {metrics['hits_3']:.6f} | "
        f"{metrics['hits_10']:.6f} | {metrics['auroc']:.6f} |"
    )


def _write(report):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "suppkg",
        "seeds": SEEDS,
        "embeddings": EMBEDDINGS,
        "protocol": PROTOCOL,
        "checkpoint_root": str(CHECKPOINT_ROOT),
        "notes_dir": str(OUT_DIR),
        "ppr_cache": str(CHECKPOINT_ROOT / "ppr_cache"),
        "results": report,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for embedding in EMBEDDINGS:
        for variant in VARIANTS:
            _write_variant(embedding, variant["name"], report[embedding][variant["name"]])
    _write_summary(report)
    print(f"Wrote {OUT_DIR / 'summary.md'}", flush=True)


def _write_variant(embedding, name, entry):
    lines = [
        f"# {entry['label']}",
        "",
        f"Embedding: `{embedding}`",
        "Dataset: suppKG (`suppkg`)",
        "",
        "Checkpoint:",
        "",
    ]
    if entry["runs"]:
        lines.append(f"`{Path(entry['runs'][0]['path']).parent.parent}`")
    lines.extend(["", "## Runs", ""])
    for item in entry["runs"]:
        lines.append(f"- seed {item['seed']}: {_format(item['metrics'])}")
        lines.append(f"  - `{item['path']}`")
    if entry["missing_seeds"]:
        lines.append(f"- seed còn thiếu: {entry['missing_seeds']}")
    lines.extend(["", "## Mean of 5 seeds", ""])
    if "mean" in entry:
        lines.append(_format(entry["mean"]))
    else:
        lines.append("Chưa đủ 5 seed để lấy mean.")
    lines.append("")
    path = OUT_DIR / f"{embedding}__{name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(report):
    lines = [
        "# model.py vs model.py + PPR",
        "",
        "Dataset: suppKG (`--data suppkg`).",
        "Seeds: 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 seed.",
        PROTOCOL,
        "",
        "Thư mục ghi chú: `compare_submodules/ppr_5seed_suppkg/`.",
        "Checkpoint gốc: `compare_submodules/checkpoints/suppkg/`.",
        "Đồ thị PPR dùng chung: `checkpoints/suppkg/ppr_cache/`.",
        "",
        "| Bản | Thư mục checkpoint |",
        "| --- | --- |",
        "| model.py | `checkpoints/suppkg/<embedding>/compare_model/seed_<seed>/` |",
        "| model.py + PPR | `checkpoints/suppkg/<embedding>/compare_model/ppr/seed_<seed>/` |",
        "",
        "File từng bản:",
        "",
    ]
    for embedding in EMBEDDINGS:
        for variant in VARIANTS:
            lines.append(f"- `{embedding}__{variant['name']}.md`")
    lines.append("")
    header = (
        "| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for embedding in EMBEDDINGS:
        lines.extend([f"## {embedding}", "", header])
        means = {}
        for variant in VARIANTS:
            entry = report[embedding][variant["name"]]
            if "mean" not in entry:
                missing = ", ".join(str(seed) for seed in entry["missing_seeds"])
                lines.append(f"| {entry['label']} | missing seeds {missing} | | | | | |")
                continue
            means[variant["name"]] = entry["mean"]
            lines.append(_table_row(entry["label"], entry["mean"]))
        if len(means) == 2:
            base = means["model"]
            ppr = means["model_plus_ppr"]
            delta = {key: ppr[key] - base[key] for key in METRIC_FIELDS}
            lines.append(_table_row("delta (PPR − model.py)", delta))
        lines.append("")
    lines.extend([
        "MR thấp hơn là tốt hơn. MRR, Hits và AUROC cao hơn là tốt hơn.",
        "Delta dương trên MRR, Hits, AUROC nghĩa là PPR cao hơn model.py.",
        "",
    ])
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON.is_file():
        raise SystemExit(f"Missing interpreter: {PYTHON}")
    for embedding in EMBEDDINGS:
        for variant in VARIANTS:
            for seed in SEEDS:
                _train(embedding, variant, seed)
                _write(_load())
    _write(_load())


if __name__ == "__main__":
    main()
