"""Retrain model.py and model_base_test.py for seeds 42-46 and save means."""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
OUT_DIR = HERE / "baseline_5seed_compare"
LOG_DIR = OUT_DIR / "logs"
CHECKPOINT_ROOT = (
    HERE / "checkpoints" / "suppkg" / "pubmedbert_pretrained_embeddings_768"
)

SEEDS = [42, 43, 44, 45, 46]
JOBS = [
    {
        "name": "model",
        "script": "train_compare_model.py",
        "metrics": CHECKPOINT_ROOT / "compare_model" / "seed_{seed}" / "metrics.json",
    },
    {
        "name": "model_base_test",
        "script": "train_compare_model_base_test.py",
        "metrics": CHECKPOINT_ROOT / "compare_model_base_test" / "baseline" / "seed_{seed}" / "metrics.json",
    },
]
METRIC_FIELDS = ["mr", "mrr", "hits_1", "hits_3", "hits_10", "auroc"]
COMMON = [
    "--data", "suppkg",
    "--iterations", "40000",
    "--evaluate_every", "1000",
    "--num_hidden_layers", "2",
    "--w", "0.75",
    "--text_embedding_file", "pubmedbert_pretrained_embeddings_768.npy",
    "--knowledge_embedding_file", "poincare_embeddings.npy",
]


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


def _train(job, seed):
    command = [str(PYTHON), "-u", str(HERE / job["script"]), "--seed", str(seed), *COMMON]
    log_path = LOG_DIR / f"{job['name']}_seed_{seed}.log"
    print(f"START {job['name']} seed={seed}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{job['name']} seed={seed} failed with code {completed.returncode}. See {log_path}"
        )
    print(f"DONE {job['name']} seed={seed}", flush=True)


def _load():
    report = {}
    for job in JOBS:
        runs = []
        missing = []
        for seed in SEEDS:
            path = Path(str(job["metrics"]).format(seed=seed))
            if not path.is_file():
                missing.append(seed)
                continue
            with path.open(encoding="utf-8") as handle:
                runs.append({"seed": seed, "metrics": _flatten(json.load(handle)), "path": str(path)})
        report[job["name"]] = {"script": job["script"], "runs": runs, "missing_seeds": missing}
    return report


def _write(report):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "seeds": SEEDS,
        "protocol": (
            "suppkg, 40000 iterations, pubmedbert_pretrained_embeddings_768, "
            "poincare, w=0.75, num_hidden_layers=2, no extra module flags"
        ),
        "models": {},
    }
    for name, payload in report.items():
        entry = {
            "script": payload["script"],
            "runs": payload["runs"],
            "missing_seeds": payload["missing_seeds"],
        }
        if len(payload["runs"]) == len(SEEDS):
            entry["mean"] = _mean([item["metrics"] for item in payload["runs"]])
        summary["models"][name] = entry
        _write_model(name, entry)
    path = OUT_DIR / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_summary(summary)
    print(f"Wrote {path}", flush=True)


def _write_model(name, entry):
    lines = [f"# {name}", "", f"Script: `{entry['script']}`", "", "## Runs", ""]
    for item in entry["runs"]:
        lines.append(f"- seed {item['seed']}: {_format(item['metrics'])}")
    if entry["missing_seeds"]:
        lines.append(f"- seed còn thiếu: {entry['missing_seeds']}")
    lines.extend(["", "## Mean", ""])
    if "mean" in entry:
        lines.append(_format(entry["mean"]))
    else:
        lines.append("Chưa đủ 5 seed để lấy mean.")
    lines.append("")
    (OUT_DIR / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary(summary):
    lines = [
        "# Baseline 5-seed means",
        "",
        "Mỗi script train 5 lần với seed 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 lần.",
        summary["protocol"] + ".",
        "",
        "| Model | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, entry in summary["models"].items():
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
    lines.extend(["", "Từng seed nằm trong file markdown của từng model.", ""])
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not PYTHON.is_file():
        raise SystemExit(f"Missing interpreter: {PYTHON}")
    for job in JOBS:
        for seed in SEEDS:
            _train(job, seed)
            _write(_load())
    _write(_load())


if __name__ == "__main__":
    main()
