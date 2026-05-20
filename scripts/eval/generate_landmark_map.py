"""Generate a first-pass landmark_map.json by fuzzy-matching segment names to KB nodes.

Usage:
    python scripts/eval/generate_landmark_map.py \
        --dataset "data/dataset_Hanoi Tourism Dataset.json" \
        --kb data/kb.json \
        --output data/eval/landmark_map.json

Output format:
    {
      "Pen Tower":    {"kb_id": "pen_tower",   "in_kb": true},
      "Banh mi":      {"kb_id": null,           "in_kb": false},
      ...
    }

After generation, manually review entries where kb_id is null or the match looks wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from difflib import get_close_matches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hanoi_caption.kb_loader import load_kb


def _collect_segment_names(dataset_path: Path) -> list[str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for video in data.get("videos", []):
        for seg in video.get("segments", []):
            name = seg.get("name", "").strip()
            if name:
                names.add(name)
    return sorted(names)


def suggest_mapping(names: list[str], kb_nodes: dict, cutoff: float = 0.55) -> dict:
    """Return {segment_name: {"kb_id": str|None, "in_kb": bool}} for each name.

    Possible outcomes:
    - in_kb=True, kb_id=<slug>: auto-matched to KB node
    - in_kb=True, kb_id=null:   no auto-match found — review and either set kb_id or set in_kb=false
    - in_kb=False, kb_id=null:  confirmed not in KB (food, craft activities, etc.)
    """
    kb_by_lower: dict[str, str] = {}  # lower(name_en) -> kb_id
    for node in kb_nodes.values():
        if node.kb_id:
            kb_by_lower[node.name_en.lower()] = node.kb_id
            # also index by kb_id slug itself for direct hits
            kb_by_lower[node.kb_id.lower().replace("_", " ")] = node.kb_id

    result: dict[str, dict] = {}
    for name in names:
        matches = get_close_matches(name.lower(), kb_by_lower.keys(), n=1, cutoff=cutoff)
        if matches:
            result[name] = {"kb_id": kb_by_lower[matches[0]], "in_kb": True}
        else:
            result[name] = {"kb_id": None, "in_kb": True}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset_Hanoi Tourism Dataset.json")
    parser.add_argument("--kb", default="data/kb.json")
    parser.add_argument("--output", default="data/eval/landmark_map.json")
    parser.add_argument("--cutoff", type=float, default=0.55, help="Fuzzy match cutoff threshold (0.0-1.0)")
    args = parser.parse_args()

    if not Path(args.dataset).exists():
        parser.error(f"Dataset not found: {args.dataset}")
    if not Path(args.kb).exists():
        parser.error(f"KB not found: {args.kb}")

    kb_nodes = load_kb(args.kb, only_objects=True)
    names = _collect_segment_names(Path(args.dataset))
    mapping = suggest_mapping(names, kb_nodes, cutoff=args.cutoff)

    Path(args.output).write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    in_kb = sum(1 for v in mapping.values() if v["in_kb"])
    print(f"Wrote {len(mapping)} entries ({in_kb} in-KB) to {args.output}")
    null_entries = [k for k, v in mapping.items() if v["kb_id"] is None and v["in_kb"]]
    if null_entries:
        print(f"WARNING: {len(null_entries)} entries have no auto-match — review and set kb_id manually OR set in_kb=false if not a landmark:")
        for n in null_entries:
            print(f"  {n!r}")


if __name__ == "__main__":
    main()
