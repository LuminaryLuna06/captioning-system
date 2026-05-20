"""Segmentation evaluation: TIoU matching, Seg-P/R/F1, LID-Acc, Refusal Rate.

Usage:
    python scripts/eval/eval_segmentation.py \
        --test-set data/eval/test_set.json \
        --results  data/eval/pipeline_results.json \
        --output   data/eval/seg_metrics.json

Output JSON schema:
    {
      "thresholds": {
        "0.3": {"precision": float, "recall": float, "f1": float, "lid_acc": float},
        "0.5": {...},
        "0.7": {...}
      },
      "refusal_rate": float,         # out-of-KB videos with 0 predicted segments
      "n_in_kb_videos": int,
      "n_out_of_kb_videos": int,
      "per_video": [...]             # per-video breakdown at threshold 0.5
    }
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def tiou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    """Temporal Intersection over Union."""
    overlap = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    if overlap == 0.0:
        return 0.0
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return overlap / union


def match_segments(
    predicted: list[dict],
    ground_truth: list[dict],
    threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching: each GT finds the best unmatched predicted.

    Returns list of (gt_idx, pred_idx, tiou_score) for matched pairs only.
    """
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gi, g in enumerate(ground_truth):
        best_score, best_pi = 0.0, -1
        for pi, p in enumerate(predicted):
            if pi in matched_pred:
                continue
            score = tiou(p["start_s"], p["end_s"], g["start_time"], g["end_time"])
            if score > best_score:
                best_score, best_pi = score, pi
        if best_score >= threshold and best_pi >= 0:
            matches.append((gi, best_pi, best_score))
            matched_pred.add(best_pi)
    return matches


def seg_metrics(
    matches: list[tuple[int, int, float]],
    n_predicted: int,
    n_gt: int,
) -> dict:
    n = len(matches)
    p = n / n_predicted if n_predicted else 0.0
    r = n / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def lid_accuracy(
    matches: list[tuple[int, int, float]],
    predicted: list[dict],
    ground_truth: list[dict],
) -> float:
    """% of matched pairs where predicted kb_id == gt kb_id."""
    if not matches:
        return 0.0
    correct = sum(
        1 for gi, pi, _ in matches
        if predicted[pi].get("kb_id") == ground_truth[gi].get("kb_id")
    )
    return round(correct / len(matches), 4)


def kb_node_precision(
    matches: list[tuple[int, int, float]],
    predicted: list[dict],
    ground_truth: list[dict],
) -> float:
    """% of matched pairs where predicted node_id == gt_node_id (MongoDB ID)."""
    if not matches:
        return 0.0
    correct = sum(
        1 for gi, pi, _ in matches
        if predicted[pi].get("node_id") == ground_truth[gi].get("gt_node_id")
    )
    return round(correct / len(matches), 4)


def evaluate_video(
    gt_segments: list[dict],
    pred_segments: list[dict],
    thresholds: list[float],
) -> dict:
    result = {}
    for thr in thresholds:
        matches = match_segments(pred_segments, gt_segments, threshold=thr)
        m = seg_metrics(matches, n_predicted=len(pred_segments), n_gt=len(gt_segments))
        m["lid_acc"] = lid_accuracy(matches, pred_segments, gt_segments)
        m["kb_node_precision"] = kb_node_precision(matches, pred_segments, gt_segments)
        result[str(thr)] = m
    return result


def run_evaluation(test_set: dict, results: list[dict]) -> dict:
    thresholds = [0.3, 0.5, 0.7]
    results_by_id = {r["video_id"]: r for r in results}
    per_video = []

    for video in test_set["in_kb"]:
        vid_id = video["video_id"]
        r = results_by_id.get(vid_id, {})
        pred = r.get("predicted_segments", [])
        gt = video["gt_segments"]
        per_video.append({
            "video_id": vid_id,
            "filename": video["filename"],
            **evaluate_video(gt, pred, thresholds),
        })

    # Aggregate across all videos per threshold
    agg: dict[str, dict] = {}
    for thr in thresholds:
        key = str(thr)
        metrics = [v[key] for v in per_video]
        agg[key] = {
            metric: round(statistics.mean(v[metric] for v in metrics), 4)
            for metric in ("precision", "recall", "f1", "lid_acc", "kb_node_precision")
        }

    # Refusal rate on out-of-KB videos
    out_of_kb = test_set.get("out_of_kb", [])
    refusals = sum(
        1 for v in out_of_kb
        if not results_by_id.get(v["video_id"], {}).get("predicted_segments")
    )
    refusal_rate = round(refusals / len(out_of_kb), 4) if out_of_kb else 0.0

    return {
        "thresholds": agg,
        "refusal_rate": refusal_rate,
        "n_in_kb_videos": len(test_set["in_kb"]),
        "n_out_of_kb_videos": len(out_of_kb),
        "per_video": per_video,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="data/eval/test_set.json")
    parser.add_argument("--results",  default="data/eval/pipeline_results.json")
    parser.add_argument("--output",   default="data/eval/seg_metrics.json")
    args = parser.parse_args()

    test_set = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    results  = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out = run_evaluation(test_set, results)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Segmentation Results:")
    for thr, m in out["thresholds"].items():
        print(f"  TIoU@{thr}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  LID={m['lid_acc']:.3f}  "
              f"KB-Node-P={m['kb_node_precision']:.3f}")
    print(f"  Refusal Rate (out-of-KB): {out['refusal_rate']:.3f}")


if __name__ == "__main__":
    main()
