"""One-command driver for Module 2.4 (backbone comparison).

For each backbone, runs:
    run_pipeline.py --skip-caption --backbone <name>
    eval_segmentation.py

Then prints a single summary table sorted by F1@0.3 desc.

Usage:
    # Run all 7 backbones, then print the table
    python scripts/eval/run_module2.py

    # Subset (still re-runs each from scratch, ~3 min per backbone)
    python scripts/eval/run_module2.py --backbones dinov3_vits16 siglip2_large

    # Skip the pipeline; just load existing seg_metrics_*.json and print the table
    python scripts/eval/run_module2.py --show-only

    # Same as above but also dump CSV for spreadsheet
    python scripts/eval/run_module2.py --show-only --csv data/eval/module2_comparison.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
PIPELINE = ROOT / "scripts" / "eval" / "run_pipeline.py"
SEGMENT  = ROOT / "scripts" / "eval" / "eval_segmentation.py"
EVAL_DIR = ROOT / "data" / "eval"
TEST_SET = EVAL_DIR / "test_set.json"

# (name, display-only params label). Add new backbones here once they exist in
# hanoi_caption.retrieval.backbones AND are registered in run_pipeline._BACKBONE_CLASSES.
BACKBONES = [
    ("siglip2_large",  "~400M"),
    ("dinov3_vitl16",  "300M"),
    ("aimv2_large",    "~300M"),
    ("dinov3_vits16",  "22M"),
    ("siglip2_base",   "~200M"),
    ("vit_base",       "86M"),
    ("resnet50",       "25M"),
]
ALL_NAMES = [n for n, _ in BACKBONES]

THRESHOLDS = ["0.3", "0.5", "0.7"]
SHORT = {"precision": "P", "recall": "R", "f1": "F1", "lid_acc": "LID"}


def run_one(name: str) -> None:
    out_results = EVAL_DIR / f"pipeline_results_{name}.json"
    out_metrics = EVAL_DIR / f"seg_metrics_{name}.json"
    print(f"\n========== {name}: pipeline ==========")
    subprocess.run(
        [
            PY, str(PIPELINE),
            "--test-set",      str(TEST_SET),
            "--video-dir",     str(ROOT / "tests" / "videos"),
            "--kb",            str(ROOT / "data" / "kb.json"),
            "--kb-images-dir", str(ROOT / "data" / "kb_images"),
            "--skip-caption",
            "--backbone",      name,
            "--output",        str(out_results),
        ],
        check=True,
    )
    print(f"========== {name}: eval ==========")
    subprocess.run(
        [
            PY, str(SEGMENT),
            "--test-set", str(TEST_SET),
            "--results",  str(out_results),
            "--output",   str(out_metrics),
        ],
        check=True,
    )


def build_table(names_in_order: list[tuple[str, str]]):
    import pandas as pd

    rows = []
    for name, params in names_in_order:
        p = EVAL_DIR / f"seg_metrics_{name}.json"
        if not p.exists():
            continue
        thr = json.loads(p.read_text(encoding="utf-8"))["thresholds"]
        row = {"params": params}
        for t in THRESHOLDS:
            for k, short in SHORT.items():
                row[f"{short}@{t}"] = round(thr[t][k], 4)
        rows.append((name, row))
    if not rows:
        return None
    df = pd.DataFrame([r for _, r in rows], index=[n for n, _ in rows])
    df.index.name = "backbone"
    return df.sort_values("F1@0.3", ascending=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--backbones", nargs="+", choices=ALL_NAMES, default=ALL_NAMES,
        help="Subset of backbones to run (default: all). Ignored with --show-only.",
    )
    parser.add_argument(
        "--show-only", action="store_true",
        help="Skip the pipeline; load existing seg_metrics_*.json and print the table.",
    )
    parser.add_argument(
        "--csv", default=None,
        help="If set, also write the summary table to this CSV path.",
    )
    args = parser.parse_args()

    if not args.show_only:
        for name in args.backbones:
            try:
                run_one(name)
            except subprocess.CalledProcessError as exc:
                print(f"\n!!! {name} FAILED (exit {exc.returncode}); continuing.\n", file=sys.stderr)

    df = build_table(BACKBONES)
    if df is None:
        print("\nNo seg_metrics_*.json found under data/eval/. Run without --show-only first.")
        return

    import pandas as pd
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n========== Module 2.4 summary (sorted by F1@0.3) ==========")
    print(df.to_string())
    print(f"\nFor a styled heatmap + bar chart view, open notebooks/03_retriever_comparison.ipynb (Section D).")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, encoding="utf-8")
        print(f"CSV -> {out}")


if __name__ == "__main__":
    main()
