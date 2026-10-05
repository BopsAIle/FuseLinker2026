"""Train Fusion V1-V10 five times each on suppkg and save the mean."""
import inspect
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
OUT_DIR = HERE / "fusion_v1_v10_compare"
LOG_DIR = OUT_DIR / "logs"
CHECKPOINT_ROOT = (
    HERE
    / "checkpoints"
    / "suppkg"
    / "pubmedbert_pretrained_embeddings_768"
    / "compare_model"
)

SEEDS = [42, 43, 44, 45, 46]
VARIANTS = [f"v{index}" for index in range(1, 11)]
METRIC_FIELDS = ["mr", "mrr", "hits_1", "hits_3", "hits_10", "auroc"]


def _metrics_path(variant, seed):
    return CHECKPOINT_ROOT / variant / f"seed_{seed}" / "metrics.json"


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
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in METRIC_FIELDS
    }


def _variant_code(variant):
    sys.path.insert(0, str(HERE))
    import fusion_v1_v10

    return inspect.getsource(fusion_v1_v10.FUSION_BUILDERS[variant])


def _train(variant, seed):
    command = [
        str(PYTHON),
        "-u",
        str(HERE / "train_compare_model.py"),
        "--data", "suppkg",
        "--seed", str(seed),
        "--iterations", "40000",
        "--evaluate_every", "1000",
        "--num_hidden_layers", "2",
        "--w", "0.75",
        "--text_embedding_file", "pubmedbert_pretrained_embeddings_768.npy",
        "--knowledge_embedding_file", "poincare_embeddings.npy",
        "--fusion_variant", variant,
    ]
    log_path = LOG_DIR / f"{variant}_seed_{seed}.log"
    print(f"START {variant} seed={seed}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{variant} seed={seed} failed with code {completed.returncode}. See {log_path}"
        )
    print(f"DONE {variant} seed={seed}", flush=True)


def _load_runs():
    report = {}
    for variant in VARIANTS:
        runs = []
        missing = []
        for seed in SEEDS:
            path = _metrics_path(variant, seed)
            if not path.is_file():
                missing.append(seed)
                continue
            with path.open(encoding="utf-8") as handle:
                runs.append({"seed": seed, "metrics": _flatten(json.load(handle))})
        report[variant] = {"runs": runs, "missing_seeds": missing}
    return report


def _format_metrics(metrics):
    return (
        f"MR {metrics['mr']:.6f} | MRR {metrics['mrr']:.6f} | "
        f"Hits@1 {metrics['hits_1']:.6f} | Hits@3 {metrics['hits_3']:.6f} | "
        f"Hits@10 {metrics['hits_10']:.6f} | AUROC {metrics['auroc']:.6f}"
    )


def _write_variant_file(variant, entry):
    lines = [
        f"# {variant}",
        "",
        "## Code",
        "",
        "```python",
        entry["code"].rstrip(),
        "```",
        "",
        "## Runs",
        "",
    ]
    if not entry["runs"] and entry["missing_seeds"]:
        lines.append(f"Chưa có kết quả. Seed còn thiếu: {entry['missing_seeds']}.")
    for item in entry["runs"]:
        lines.append(f"- seed {item['seed']}: {_format_metrics(item['metrics'])}")
    if entry["missing_seeds"] and entry["runs"]:
        lines.append(f"- seed còn thiếu: {entry['missing_seeds']}")
    lines.extend(["", "## Mean", ""])
    if "mean" in entry:
        lines.append(_format_metrics(entry["mean"]))
    else:
        lines.append("Chưa đủ 5 seed để lấy mean.")
    lines.append("")
    (OUT_DIR / f"{variant}.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_md(summary):
    ranked = []
    for name, entry in summary["variants"].items():
        mean_mrr = entry["mean"]["mrr"] if "mean" in entry else None
        ranked.append((mean_mrr is None, -(mean_mrr or 0.0), name, entry))
    ranked.sort()
    lines = [
        "# Fusion V1-V10 means on suppkg",
        "",
        "Mỗi bản train 5 lần với seed 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 lần.",
        "Script: train_compare_model.py. Protocol: suppkg, 40000 iterations, pubmedbert_pretrained_embeddings_768, poincare, w=0.75, num_hidden_layers=2.",
        "",
        "| Variant | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, _, name, entry in ranked:
        if "mean" not in entry:
            missing = ", ".join(str(seed) for seed in entry["missing_seeds"])
            lines.append(f"| {name} | missing seeds {missing} | | | | | |")
            continue
        mean = entry["mean"]
        lines.append(
            f"| {name} | {mean['mr']:.6f} | {mean['mrr']:.6f} | "
            f"{mean['hits_1']:.6f} | {mean['hits_3']:.6f} | "
            f"{mean['hits_10']:.6f} | {mean['auroc']:.6f} |"
        )
    lines.extend(["", "Năm lần chạy và code của từng bản nằm trong file markdown cùng thư mục.", ""])
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(report):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"seeds": SEEDS, "variants": {}}
    for variant, payload in report.items():
        runs = payload["runs"]
        entry = {
            "code": _variant_code(variant),
            "runs": runs,
            "missing_seeds": payload["missing_seeds"],
        }
        if len(runs) == len(SEEDS):
            entry["mean"] = _mean([item["metrics"] for item in runs])
        summary["variants"][variant] = entry
        _write_variant_file(variant, entry)

    summary_path = OUT_DIR / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    _write_summary_md(summary)
    print(f"Wrote {summary_path}", flush=True)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON.is_file():
        raise SystemExit(f"Missing interpreter: {PYTHON}")
    for variant in VARIANTS:
        for seed in SEEDS:
            if _metrics_path(variant, seed).is_file():
                print(f"SKIP {variant} seed={seed}", flush=True)
                continue
            _train(variant, seed)
            _write_outputs(_load_runs())
    _write_outputs(_load_runs())


if __name__ == "__main__":
    main()
