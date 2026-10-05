"""Train Fusion V1-V10 from 10_fusion_variants.md, three seeds each."""
import json

import run_fusion_variants as sweep

sweep.OUT_DIR = sweep.HERE / "fusion_v10_compare"
sweep.LOG_DIR = sweep.OUT_DIR / "logs"
sweep.VARIANTS = [f"v{index}" for index in range(1, 11)]
PREVIOUS_SUMMARY = sweep.HERE / "fusion_gate_compare" / "summary.json"


def _fixed_reference():
    if not PREVIOUS_SUMMARY.is_file():
        return None
    with PREVIOUS_SUMMARY.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    entry = payload.get("variants", {}).get("fixed", {})
    return entry.get("mean")


def _write_summary_md(summary):
    ranked = []
    for name, entry in summary["variants"].items():
        mean_mrr = entry["mean"]["mrr"] if "mean" in entry else None
        ranked.append((mean_mrr is None, -(mean_mrr or 0.0), name, entry))
    ranked.sort()
    lines = [
        "# Fusion V1-V10 means",
        "",
        "Mỗi bản train 3 lần với seed 42, 43, 44. Mean là trung bình cộng của 3 lần.",
        "Nguồn code: compare_submodules/10_fusion_variants.md.",
        "Protocol: suppkg, 40000 iterations, bert_pretrained_embeddings_768, poincare, w=0.75, num_hidden_layers=2.",
        "",
        "| Variant | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    fixed_mean = _fixed_reference()
    if fixed_mean is not None:
        lines.append(
            f"| fixed (lần trước) | {fixed_mean['mr']:.6f} | {fixed_mean['mrr']:.6f} | "
            f"{fixed_mean['hits_1']:.6f} | {fixed_mean['hits_3']:.6f} | "
            f"{fixed_mean['hits_10']:.6f} | {fixed_mean['auroc']:.6f} |"
        )
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
    lines.extend(["", "Chi tiết từng seed và code nằm trong file markdown của từng bản.", ""])
    (sweep.OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


sweep._write_summary_md = _write_summary_md


if __name__ == "__main__":
    sweep.main()
