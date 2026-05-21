"""LLM-as-judge: GPT-4o evaluates each caption on 4 criteria (1-5 scale).

Usage:
    python scripts/eval/eval_llm_judge.py \
        --test-set   data/eval/test_set.json \
        --results    data/eval/pipeline_results.json \
        --output     data/eval/llm_scores.json \
        --api-key    $OPENAI_API_KEY

Output JSON schema (list):
    [
      {
        "video_id": str,
        "landmark_name": str,
        "kb_id": str,
        "scores": {"factual_accuracy": int, "visual_grounding": int, "tone": int, "hallucination": int},
        "reasoning": str
      }
    ]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

JUDGE_SYSTEM = (
    "You are an expert evaluator of AI-generated travel guide content. "
    "Be objective and consistent. Respond only with the JSON object requested."
)

JUDGE_USER = """\
Evaluate this AI-generated tour-guide caption for a Hanoi landmark video segment.

Landmark: {name_en}

Knowledge base facts provided to the AI:
{kb_description}

AI-generated caption:
{predicted_caption}

Reference caption (human-written):
{reference_caption}

Rate the AI caption on each criterion from 1 (poor) to 5 (excellent):
- factual_accuracy: Do all stated facts come from the knowledge base or direct visual observation?
- visual_grounding: Does the caption describe what is actually visible in the video, rather than only repeating KB facts?
- tone: Is the writing warm, engaging, and in a tour-guide voice (not a list or encyclopedia entry)?
- hallucination: Are there invented facts not found in the KB? (5 = no hallucination, 1 = many invented facts)

Respond with ONLY this JSON object:
{{"factual_accuracy": X, "visual_grounding": X, "tone": X, "hallucination": X, "reasoning": "one sentence"}}\
"""

_SCORE_KEYS = ("factual_accuracy", "visual_grounding", "tone", "hallucination")


def build_prompt(item: dict) -> str:
    return JUDGE_USER.format(
        name_en=item["name_en"],
        kb_description=item["kb_description"][:2000],
        predicted_caption=item["predicted_caption"],
        reference_caption=item["reference_caption"],
    )


def parse_response(content: str) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON from judge: {exc}\nContent: {content[:200]}") from exc
    missing = [k for k in _SCORE_KEYS if k not in data]
    if missing:
        raise ValueError(f"invalid JSON from judge: missing keys {missing}\nContent: {content[:200]}")
    for k in _SCORE_KEYS:
        data[k] = max(1, min(5, int(data[k])))
    return data


def judge_caption(client, item: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": build_prompt(item)},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return parse_response(response.choices[0].message.content)


def _collect_items(test_set: dict, results: list[dict]) -> list[dict]:
    """Collect matched + correct-landmark pairs for LLM judging."""
    from scripts.eval.eval_segmentation import match_segments
    results_by_id = {r["video_id"]: r for r in results}
    items = []
    for video in test_set.get("in_kb", []):
        vid_id = video["video_id"]
        r = results_by_id.get(vid_id, {})
        pred_segs = r.get("predicted_segments", [])
        gt_segs   = video["gt_segments"]
        if not pred_segs or not gt_segs:
            continue
        matches = match_segments(pred_segs, gt_segs, threshold=0.5)
        for gi, pi, _ in matches:
            gt   = gt_segs[gi]
            pred = pred_segs[pi]
            if pred.get("kb_id") != gt.get("kb_id"):
                continue
            items.append({
                "video_id": vid_id,
                "landmark_name": gt["landmark_name"],
                "kb_id": gt["kb_id"],
                "name_en": pred.get("name_en", gt.get("landmark_name", "")),
                "kb_description": gt.get("kb_description", ""),
                "predicted_caption": pred.get("caption", ""),
                "reference_caption": gt.get("reference_caption", ""),
            })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set",    default="data/eval/test_set.json")
    parser.add_argument("--results",     default="data/eval/pipeline_results.json")
    parser.add_argument("--output",      default="data/eval/llm_scores.json")
    parser.add_argument("--api-key",     default=os.environ.get("OPENAI_API_KEY", ""))
    args = parser.parse_args()

    import openai
    client = openai.OpenAI(api_key=args.api_key)

    test_set = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    results  = json.loads(Path(args.results).read_text(encoding="utf-8"))

    items = _collect_items(test_set, results)
    print(f"Judging {len(items)} segments with GPT-4o...")

    scored = []
    for i, item in enumerate(items, 1):
        try:
            scores = judge_caption(client, item)
            scored.append({
                "video_id": item["video_id"],
                "landmark_name": item["landmark_name"],
                "kb_id": item["kb_id"],
                "scores": {k: scores[k] for k in _SCORE_KEYS},
                "reasoning": scores.get("reasoning", ""),
            })
            print(f"  [{i}/{len(items)}] {item['landmark_name']}: "
                  f"FA={scores['factual_accuracy']} VG={scores['visual_grounding']} "
                  f"T={scores['tone']} H={scores['hallucination']}")
        except Exception as exc:
            import traceback
            print(f"  [{i}/{len(items)}] ERROR for {item['video_id']}: {exc}")
            if not isinstance(exc, ValueError):
                traceback.print_exc()
        time.sleep(0.5)  # respect rate limits

    Path(args.output).write_text(
        json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(scored)} scores -> {args.output}")


if __name__ == "__main__":
    main()
