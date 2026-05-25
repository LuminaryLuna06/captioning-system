"""Extract one representative frame per landmark into tests/fixtures/retriever_frames/
for use as fixed query images by notebooks/03_retriever_comparison.ipynb.

Run once after pulling videos into tests/videos/:

    python scripts/data_collection/extract_fixed_frames.py
        [--video-dir tests/videos]
        [--out-dir tests/fixtures/retriever_frames]

Hard-coded (video_filename, timestamp_s, kb_id) triples — edit the list below
to add or remove landmarks.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (video filename under --video-dir, timestamp in seconds, kb_id to save as).
# The kb_id strings are only used as the saved filename + the "expected" label
# in the notebook's top-K grid; they DO NOT need to match real kb_ids in
# data/kb.json. Update them to match if you want the grid title to read as a
# proper kb_id (run `ls data/kb_images/` to see the canonical names).
FIXED_FRAMES = [
    ("NhaThoLon_S_T03.MOV",                  10.0, "nha_tho_lon"),
    ("NhaHatLon_S_T04.MOV",                  10.0, "nha_hat_lon"),
    ("NhaKhachChinhPhu_S_T02.MOV",           10.0, "nha_khach_chinh_phu"),
    ("A1_018_DenNgocSonToanCanh_M_T02.mp4",  10.0, "den_ngoc_son"),
    ("FLN_BaoTangGom_T48_S.MOV",             10.0, "bao_tang_gom"),
]


def extract_one(video_path: Path, timestamp_s: float):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"unreadable FPS for {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp_s * fps)))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame at t={timestamp_s}s from {video_path}")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", default="tests/videos", type=Path)
    p.add_argument("--out-dir", default="tests/fixtures/retriever_frames", type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("extract_fixed_frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for filename, t, kb_id in FIXED_FRAMES:
        video_path = args.video_dir / filename
        if not video_path.exists():
            log.warning("missing: %s", video_path)
            continue
        try:
            img = extract_one(video_path, t)
        except Exception as e:
            log.warning("failed %s (%s): %s", filename, kb_id, e)
            continue
        out_path = args.out_dir / f"{kb_id}.jpg"
        img.save(out_path, "JPEG", quality=92)
        log.info("wrote %s", out_path)
        n_ok += 1

    log.info("extracted %d/%d fixed frames", n_ok, len(FIXED_FRAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
