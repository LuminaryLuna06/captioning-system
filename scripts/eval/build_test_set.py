"""Build the evaluation test set from the annotated dataset.

Usage:
    python scripts/eval/build_test_set.py \
        --dataset "data/dataset_Hanoi Tourism Dataset.json" \
        --landmark-map data/eval/landmark_map.json \
        --output data/eval/test_set.json \
        --total 30 \
        --seed 42

Output JSON schema:
    {
      "in_kb": [
        {
          "video_id": str,
          "filename": str,
          "duration": float,
          "gt_segments": [
            {
              "start_time": float,
              "end_time": float,
              "landmark_name": str,
              "kb_id": str,
              "gt_node_id": str,        # first knowledge_base_ids entry (MongoDB ID)
              "reference_caption": str,
              "kb_description": str     # concatenated knowledge_base_items descriptions
            }
          ]
        }
      ],
      "out_of_kb": [
        {"video_id": str, "filename": str, "duration": float}
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _first_combined_caption(regions: list[dict]) -> str:
    for r in regions:
        cap = r.get("captions", {}).get("en", {}).get("combined", "").strip()
        if cap:
            return cap
    return ""


def _kb_description(regions: list[dict]) -> str:
    texts = []
    for r in regions:
        for item in r.get("knowledge_base_items", []):
            desc = item.get("description", "").strip()
            if desc:
                texts.append(desc)
    return "\n\n".join(texts)


def _gt_node_id(regions: list[dict]) -> str:
    for r in regions:
        ids = r.get("knowledge_base_ids", [])
        if ids:
            return ids[0]
    return ""


def build_test_set(
    dataset_path: "Path | str",
    landmark_map_path: "Path | str",
    total_videos: int = 30,
    seed: int = 42,
) -> dict:
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    lmap: dict[str, dict] = json.loads(Path(landmark_map_path).read_text(encoding="utf-8"))

    in_kb_by_kb_id: dict[str, list[dict]] = defaultdict(list)
    out_of_kb: list[dict] = []

    for video in data.get("videos", []):
        video_id = video["id"]
        filename = video.get("filename", "")
        duration = video.get("duration", 0.0)
        good_segs = []
        has_out_of_kb = False

        for seg in video.get("segments", []):
            name = seg.get("name", "").strip()
            combined = _first_combined_caption(seg.get("regions", []))
            if not combined:
                continue  # no annotation

            entry = lmap.get(name, {})
            kb_id = entry.get("kb_id")
            in_kb = entry.get("in_kb", False)

            if in_kb and kb_id:
                good_segs.append({
                    "start_time": seg.get("start_time", 0.0),
                    "end_time": seg.get("end_time", duration),
                    "landmark_name": name,
                    "kb_id": kb_id,
                    "gt_node_id": _gt_node_id(seg.get("regions", [])),
                    "reference_caption": combined,
                    "kb_description": _kb_description(seg.get("regions", [])),
                })
            else:
                has_out_of_kb = True

        if good_segs:
            for seg in good_segs:
                in_kb_by_kb_id[seg["kb_id"]].append({
                    "video_id": video_id,
                    "filename": filename,
                    "duration": duration,
                    "gt_segments": [seg],
                })
        if has_out_of_kb and not good_segs:
            out_of_kb.append({"video_id": video_id, "filename": filename, "duration": duration})

    selected_in_kb = _stratified_sample(in_kb_by_kb_id, total_videos, seed)

    return {"in_kb": selected_in_kb, "out_of_kb": out_of_kb}


def _stratified_sample(
    by_kb_id: dict[str, list[dict]], total: int, seed: int
) -> list[dict]:
    rng = random.Random(seed)
    kb_ids = sorted(by_kb_id.keys())
    if not kb_ids:
        return []
    per_lm = max(1, total // len(kb_ids))
    selected: list[dict] = []
    for kb_id in kb_ids:
        pool = by_kb_id[kb_id]
        n = min(per_lm, len(pool))
        selected.extend(rng.sample(pool, n))
        if len(selected) >= total:
            break
    # top-up if still under total
    selected_set = {id(v) for v in selected}
    remaining = [v for vlist in by_kb_id.values() for v in vlist if id(v) not in selected_set]
    if len(selected) < total and remaining:
        extra = rng.sample(remaining, min(total - len(selected), len(remaining)))
        selected.extend(extra)
    return selected[:total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_Hanoi Tourism Dataset.json")
    parser.add_argument("--landmark-map", default="data/eval/landmark_map.json")
    parser.add_argument("--output", default="data/eval/test_set.json")
    parser.add_argument("--total", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_test_set(args.dataset, args.landmark_map, args.total, args.seed)
    Path(args.output).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    landmarks = {s["kb_id"] for v in result["in_kb"] for s in v["gt_segments"]}
    print(f"in_kb: {len(result['in_kb'])} videos, {len(landmarks)} unique landmarks")
    print(f"out_of_kb: {len(result['out_of_kb'])} videos")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
