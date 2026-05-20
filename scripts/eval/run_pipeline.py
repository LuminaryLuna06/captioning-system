"""Run caption_video() on every in-KB video in the test set.

Usage:
    python scripts/eval/run_pipeline.py \
        --test-set  data/eval/test_set.json \
        --video-dir /path/to/video/folder \
        --output    data/eval/pipeline_results.json \
        --dino-index data/cache/dino_faiss.index \
        --id-map     data/cache/id_map.json \
        --sample-fps 1.0 \
        --resume                   # skip videos already in output file

Output JSON schema (list):
    [
      {
        "video_id": str,
        "filename": str,
        "predicted_segments": [
          {"start_s": float, "end_s": float, "kb_id": str, "node_id": str,
           "name_en": str, "confidence": float, "caption": str}
        ],
        "error": str | null
      }
    ]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hanoi_caption.kb_loader import load_kb
from hanoi_caption.video_pipeline import caption_video


def _find_video(filename: str, video_dir: Path) -> Path | None:
    p = video_dir / filename
    if p.exists():
        return p
    stem = Path(filename).stem.lower()
    for f in video_dir.iterdir():
        if f.stem.lower() == stem:
            return f
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set",   default="data/eval/test_set.json")
    parser.add_argument("--video-dir",  required=True,
                        help="Root folder that contains the .MOV/.MP4 files")
    parser.add_argument("--output",     default="data/eval/pipeline_results.json")
    parser.add_argument("--kb",         default="data/kb.json")
    parser.add_argument("--dino-index", default="data/cache/dino_faiss.index")
    parser.add_argument("--id-map",     default="data/cache/id_map.json")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--resume",     action="store_true",
                        help="Load existing output and skip already-processed videos")
    args = parser.parse_args()

    test_set  = json.loads(Path(args.test_set).read_text(encoding="utf-8"))
    kb_nodes  = load_kb(args.kb, only_objects=True)
    video_dir = Path(args.video_dir)

    all_videos = test_set.get("in_kb", []) + test_set.get("out_of_kb", [])

    results: list[dict] = []
    done_ids: set[str] = set()
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        done_ids = {r["video_id"] for r in results}
        print(f"Resuming: {len(done_ids)} videos already done")

    for i, video in enumerate(all_videos, 1):
        vid_id   = video["video_id"]
        filename = video["filename"]
        if vid_id in done_ids:
            continue

        video_path = _find_video(filename, video_dir)
        if video_path is None:
            print(f"[{i}/{len(all_videos)}] SKIP (file not found): {filename}")
            results.append({"video_id": vid_id, "filename": filename,
                            "predicted_segments": [], "error": "file_not_found"})
            out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            continue

        print(f"[{i}/{len(all_videos)}] {filename} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            segs = caption_video(
                video_path=video_path,
                kb_nodes=kb_nodes,
                dino_index_path=args.dino_index,
                id_map_path=args.id_map,
                sample_fps=args.sample_fps,
            )
            elapsed = time.perf_counter() - t0
            pred = [s.model_dump(exclude={"debug"}) for s in segs]
            results.append({"video_id": vid_id, "filename": filename,
                            "predicted_segments": pred, "error": None})
            print(f"{len(segs)} seg(s) in {elapsed:.1f}s")
        except Exception as exc:
            import traceback
            elapsed = time.perf_counter() - t0
            results.append({"video_id": vid_id, "filename": filename,
                            "predicted_segments": [], "error": str(exc)})
            print(f"ERROR in {elapsed:.1f}s: {exc}")
            traceback.print_exc()

        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    n_ok  = sum(1 for r in results if r["error"] is None)
    n_miss = sum(1 for r in results if r["error"] == "file_not_found")
    n_err = sum(1 for r in results if r["error"] not in (None, "file_not_found"))
    print(f"\nDone: {n_ok} ok, {n_miss} file-not-found, {n_err} other errors")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
