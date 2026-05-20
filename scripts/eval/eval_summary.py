"""Aggregate all eval outputs into a summary table for the paper.

Usage:
    python scripts/eval/eval_summary.py \
        --seg-metrics   data/eval/seg_metrics.json \
        --caption-metrics data/eval/caption_metrics.json \
        --llm-scores    data/eval/llm_scores.json \
        --output-csv    data/eval/eval_summary.csv \
        --output-latex  data/eval/eval_summary.tex
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def aggregate_scores(scores: list[dict], keys: list[str]) -> dict:
    """Compute mean, std, n for each key across a list of score dicts.

    Skips entries where a key is missing.
    """
    result: dict[str, dict] = {}
    for k in keys:
        vals = [s[k] for s in scores if k in s]
        if vals:
            result[k] = {
                "mean": round(statistics.mean(vals), 4),
                "std":  round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "n":    len(vals),
            }
    return result


def format_latex_row(label: str, stats: dict) -> str:
    return rf"{label} & {stats['mean']:.4f} $\pm$ {stats['std']:.4f} \\"


def _llm_scores_flat(llm_scores: list[dict]) -> list[dict]:
    return [
        {**item.get("scores", {}), "reasoning": item.get("reasoning", "")}
        for item in llm_scores
        if "scores" in item
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-metrics",      default="data/eval/seg_metrics.json")
    parser.add_argument("--caption-metrics",  default="data/eval/caption_metrics.json")
    parser.add_argument("--llm-scores",       default="data/eval/llm_scores.json")
    parser.add_argument("--output-csv",       default="data/eval/eval_summary.csv")
    parser.add_argument("--output-latex",     default="data/eval/eval_summary.tex")
    args = parser.parse_args()

    seg     = json.loads(Path(args.seg_metrics).read_text(encoding="utf-8"))
    cap     = json.loads(Path(args.caption_metrics).read_text(encoding="utf-8"))
    llm_raw = json.loads(Path(args.llm_scores).read_text(encoding="utf-8"))
    llm     = _llm_scores_flat(llm_raw)

    # --- Segmentation summary (aggregated, no per-segment std) ---
    seg_thr = seg["thresholds"]
    seg_rows = []
    for thr in ["0.3", "0.5", "0.7"]:
        m = seg_thr.get(thr, {})
        seg_rows.append({"metric": f"Seg-F1@{thr}",   "mean": m.get("f1", 0.0),       "std": 0.0})
        seg_rows.append({"metric": f"LID-Acc@{thr}",  "mean": m.get("lid_acc", 0.0),  "std": 0.0})
    seg_rows.append({"metric": "Refusal Rate", "mean": seg.get("refusal_rate", 0.0), "std": 0.0})

    # --- Caption summary (with per-segment std) ---
    per_seg = cap.get("per_segment", [])
    cap_agg = aggregate_scores(per_seg, ["bleu4", "rouge_l", "meteor", "bertscore_f1", "kb_coverage"])
    cap_rows = [
        {"metric": k.upper().replace("_", "-"), **v}
        for k, v in cap_agg.items()
    ]

    # --- LLM judge summary (with per-segment std) ---
    llm_agg = aggregate_scores(llm, ["factual_accuracy", "visual_grounding", "tone", "hallucination"])
    llm_rows = [
        {"metric": k.replace("_", " ").title(), **v}
        for k, v in llm_agg.items()
    ]

    all_rows = seg_rows + cap_rows + llm_rows

    # Print to console
    print(f"\n{'Metric':<25} {'Mean':>8} {'Std':>8} {'N':>5}")
    print("-" * 50)
    for r in all_rows:
        n = r.get("n", "-")
        print(f"  {r['metric']:<23} {r['mean']:>8.4f} {r.get('std', 0.0):>8.4f} {n!s:>5}")

    # CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nCSV  -> {args.output_csv}")

    # LaTeX
    latex_lines = [
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"\textbf{Metric} & \textbf{Mean} & \textbf{Std} \\",
        r"\hline",
    ]
    for r in all_rows:
        latex_lines.append(
            rf"{r['metric']} & {r['mean']:.4f} & {r.get('std', 0.0):.4f} \\"
        )
    latex_lines += [r"\hline", r"\end{tabular}"]
    Path(args.output_latex).write_text("\n".join(latex_lines), encoding="utf-8")
    print(f"LaTeX -> {args.output_latex}")


if __name__ == "__main__":
    main()
