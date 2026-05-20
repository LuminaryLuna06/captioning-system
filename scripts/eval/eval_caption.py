"""Caption quality evaluation: BLEU-4, ROUGE-L, METEOR, BERTScore-F1, KB_Coverage.

Usage:
    python scripts/eval/eval_caption.py \
        --test-set  data/eval/test_set.json \
        --results   data/eval/pipeline_results.json \
        --output    data/eval/caption_metrics.json

Only evaluates captions for matched + correct-landmark segments (TIoU@0.5, LID correct).

Output JSON schema:
    {
      "overall": {"bleu4": float, "rouge_l": float, "meteor": float,
                  "bertscore_f1": float, "kb_coverage": float},
      "per_segment": [
        {"video_id": str, "landmark_name": str, "kb_id": str,
         "bleu4": float, "rouge_l": float, "meteor": float,
         "bertscore_f1": float, "kb_coverage": float}
      ],
      "n_evaluated": int
    }
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "it", "its", "this", "that",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "and", "or",
    "but", "has", "have", "been", "be", "as", "which", "who", "also", "most",
    "more", "very", "their", "they", "these", "those", "into", "through",
    "during", "before", "after", "both", "each", "than", "only", "over",
    "such", "will", "would", "could", "should",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) >= 4 and w not in _STOPWORDS}


def kb_coverage(kb_text: str, caption: str) -> float:
    """Fraction of KB keywords that appear in the caption."""
    kb_kw = _keywords(kb_text)
    if not kb_kw:
        return 0.0
    cap_kw = _keywords(caption)
    return round(len(kb_kw & cap_kw) / len(kb_kw), 4)


def caption_metrics_one(predicted: str, reference: str) -> dict:
    """Compute BLEU-4, ROUGE-L, METEOR for a single (pred, ref) pair."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rs

    ref_tokens = reference.lower().split()
    pred_tokens = predicted.lower().split()

    bleu4 = sentence_bleu(
        [ref_tokens], pred_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=SmoothingFunction().method1,
    )
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(reference, predicted)["rougeL"].fmeasure
    meteor = meteor_score([ref_tokens], pred_tokens)

    return {
        "bleu4":   round(bleu4, 4),
        "rouge_l": round(rouge_l, 4),
        "meteor":  round(meteor, 4),
    }


def bertscore_batch(predictions: list[str], references: list[str]) -> list[float]:
    """Batch BERTScore-F1 using roberta-large. Returns one score per pair."""
    if not predictions:
        return []
    from bert_score import score
    _, _, F1 = score(
        predictions, references,
        lang="en", model_type="roberta-large", verbose=False,
    )
    return [round(f.item(), 4) for f in F1]


def _collect_eval_pairs(test_set: dict, results: list[dict]) -> list[dict]:
    """Return matched + correct-landmark pairs for caption evaluation."""
    results_by_id = {r["video_id"]: r for r in results}
    pairs = []

    from scripts.eval.eval_segmentation import match_segments

    for video in test_set.get("in_kb", []):
        vid_id = video["video_id"]
        r = results_by_id.get(vid_id, {})
        pred_segs = r.get("predicted_segments", [])
        gt_segs   = video["gt_segments"]
        if not pred_segs or not gt_segs:
            continue
        matches = match_segments(pred_segs, gt_segs, threshold=0.5)
        for gi, pi, _ in matches:
            gt = gt_segs[gi]
            pred = pred_segs[pi]
            if pred.get("kb_id") != gt.get("kb_id"):
                continue  # wrong landmark — skip caption eval
            pairs.append({
                "video_id": vid_id,
                "landmark_name": gt["landmark_name"],
                "kb_id": gt["kb_id"],
                "predicted_caption": pred.get("caption", ""),
                "reference_caption": gt.get("reference_caption", ""),
                "kb_description": gt.get("kb_description", ""),
            })
    return pairs


def run_evaluation(test_set: dict, results: list[dict]) -> dict:
    pairs = _collect_eval_pairs(test_set, results)
    if not pairs:
        return {"overall": {}, "per_segment": [], "n_evaluated": 0}

    per_seg = []
    for p in pairs:
        m = caption_metrics_one(p["predicted_caption"], p["reference_caption"])
        m["kb_coverage"] = kb_coverage(p["kb_description"], p["predicted_caption"])
        per_seg.append({
            "video_id": p["video_id"],
            "landmark_name": p["landmark_name"],
            "kb_id": p["kb_id"],
            **m,
        })

    # BERTScore in one batch
    preds = [p["predicted_caption"] for p in pairs]
    refs  = [p["reference_caption"]  for p in pairs]
    bert_scores = bertscore_batch(preds, refs)
    for seg, bs in zip(per_seg, bert_scores):
        seg["bertscore_f1"] = bs

    keys = ["bleu4", "rouge_l", "meteor", "bertscore_f1", "kb_coverage"]
    overall = {
        k: round(statistics.mean(s[k] for s in per_seg), 4)
        for k in keys
    }
    return {"overall": overall, "per_segment": per_seg, "n_evaluated": len(per_seg)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set",    default="data/eval/test_set.json")
    parser.add_argument("--results",     default="data/eval/pipeline_results.json")
    parser.add_argument("--output",      default="data/eval/caption_metrics.json")
    args = parser.parse_args()

    test_set = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    results  = json.loads(Path(args.results).read_text(encoding="utf-8"))

    out = run_evaluation(test_set, results)
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Caption Metrics (n={out['n_evaluated']} segments):")
    for k, v in out["overall"].items():
        print(f"  {k:15s}: {v:.4f}")


if __name__ == "__main__":
    main()
