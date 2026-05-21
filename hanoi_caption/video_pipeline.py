"""Video captioning pipeline: DINOv3 retrieval per frame + DAM multi-frame caption per segment.

See `docs/superpowers/specs/2026-05-14-video-caption-pipeline-design.md`.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hanoi_caption.kb_loader import index_by_kb_id
from hanoi_caption.model_registry import registry
from hanoi_caption.region_describer import MODEL_NAME as DAM_NAME
from hanoi_caption.schemas import KBNode, VideoSegment

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRecord:
    timestamp_s: float
    kb_id: str | None      # None = unknown / low-confidence
    score: float


def _majority_vote(records: list[FrameRecord], window: int) -> list[str | None]:
    """Replace each frame's kb_id with the majority over a sliding window centred on it.

    Edges use whatever window is available (no padding). Ties keep the original frame's kb_id.
    """
    if window <= 1:
        return [r.kb_id for r in records]
    half = window // 2
    out: list[str | None] = []
    for i, r in enumerate(records):
        lo = max(0, i - half)
        hi = min(len(records), i + half + 1)
        votes = [records[j].kb_id for j in range(lo, hi)]
        counts = Counter(votes)
        top_count = max(counts.values())
        winners = [k for k, c in counts.items() if c == top_count]
        if r.kb_id in winners:
            out.append(r.kb_id)        # tie-break: keep self if tied
        else:
            out.append(winners[0])
    return out


def _runs(records: list[FrameRecord], smoothed: list[str | None],
          stride_s: float) -> list[dict[str, Any]]:
    """Merge consecutive frames with the same smoothed kb_id into runs."""
    runs: list[dict[str, Any]] = []
    i = 0
    n = len(records)
    while i < n:
        j = i
        while j + 1 < n and smoothed[j + 1] == smoothed[i]:
            j += 1
        frame_indices = list(range(i, j + 1))
        scores = [records[k].score for k in frame_indices]
        runs.append({
            "kb_id": smoothed[i],
            "start_s": records[i].timestamp_s,
            "end_s": records[j].timestamp_s + stride_s,
            "frame_indices": frame_indices,
            "confidence": sum(scores) / len(scores),
        })
        i = j + 1
    return runs


def _merge_adjacent(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive runs that share the same kb_id into a single run."""
    merged: list[dict[str, Any]] = []
    for run in runs:
        if merged and merged[-1]["kb_id"] == run["kb_id"]:
            prev = merged[-1]
            new_indices = prev["frame_indices"] + run["frame_indices"]
            n_prev = len(prev["frame_indices"])
            n_cur = len(run["frame_indices"])
            n_total = n_prev + n_cur
            merged[-1] = {
                "kb_id": prev["kb_id"],
                "start_s": prev["start_s"],
                "end_s": run["end_s"],
                "frame_indices": new_indices,
                "confidence": (prev["confidence"] * n_prev + run["confidence"] * n_cur) / n_total,
            }
        else:
            merged.append(run)
    return merged


def _absorb_short_runs(runs: list[dict[str, Any]], min_seconds: float) -> list[dict[str, Any]]:
    """Absorb short runs into the longer neighbor.

    Rules:
    - Unknown (kb_id is None) runs are NEVER absorbed into a known neighbor; they are
      simply dropped at the final filter step. This function relabels a short known
      run to its longer neighbor's kb_id (after which `_merge_adjacent` collapses
      them into one run), OR drops the short run by relabeling to None when no known
      neighbor exists.
    - Among two known neighbors, absorb into the longer (in duration). Ties go to the
      preceding neighbor.
    """
    if not runs:
        return runs

    # Repeatedly find a short known run and try to absorb it. Stop when no more changes.
    # After each relabeling, immediately merge adjacent same-kb_id runs so that the merged
    # run has an updated (longer) duration — preventing an infinite loop where a short run
    # is relabeled to match a neighbor but stays short because the merge hasn't happened yet.
    changed = True
    while changed:
        changed = False
        runs = _merge_adjacent(runs)          # merge before scanning so durations are current
        for i, run in enumerate(runs):
            if run["kb_id"] is None:
                continue
            duration = run["end_s"] - run["start_s"]
            if duration >= min_seconds:
                continue
            # candidate neighbors: nearest *known* runs on either side
            left = next((runs[k] for k in range(i - 1, -1, -1) if runs[k]["kb_id"] is not None), None)
            right = next((runs[k] for k in range(i + 1, len(runs)) if runs[k]["kb_id"] is not None), None)
            target = None
            if left and right:
                ld = left["end_s"] - left["start_s"]
                rd = right["end_s"] - right["start_s"]
                target = left if ld >= rd else right        # tie -> left
            elif left:
                target = left
            elif right:
                target = right
            if target is None:
                # No known neighbor at all -> drop by relabeling to unknown.
                runs[i] = {**run, "kb_id": None}
                changed = True
                break
            # Absorb: relabel this run to target's kb_id; next loop iteration will merge.
            runs[i] = {**run, "kb_id": target["kb_id"]}
            changed = True
            break

    # Final merge pass to combine any remaining adjacent same-kb_id runs.
    return _merge_adjacent(runs)


def smooth_and_group(records: list[FrameRecord], *, smooth_window: int,
                     min_segment_seconds: float, stride_s: float) -> list[dict[str, Any]]:
    """Smooth per-frame kb_ids, merge into runs, absorb short runs, drop unknowns.

    Returns a list of segment dicts with keys:
        kb_id, start_s, end_s, frame_indices, confidence
    Unknown (kb_id is None) segments are excluded from the result.
    """
    if not records:
        return []
    smoothed = _majority_vote(records, smooth_window)
    runs = _runs(records, smoothed, stride_s)
    runs = _absorb_short_runs(runs, min_seconds=min_segment_seconds)
    return [r for r in runs if r["kb_id"] is not None]


def pick_frame_indices(*, segment_seconds: float, available_indices: list[int],
                       budget: tuple[int, int]) -> list[int]:
    """Select evenly-spaced frame indices for a single DAM call.

    target_K = clamp(ceil(segment_seconds), min=budget[0], max=budget[1])
    actual_K = min(target_K, len(available_indices))   # no upsampling
    """
    lo, hi = budget
    if lo > hi or lo < 1:
        raise ValueError(f"invalid frame budget: {budget}")
    if not available_indices:
        return []
    target_k = min(max(math.ceil(segment_seconds), lo), hi)
    actual_k = min(target_k, len(available_indices))
    if actual_k == len(available_indices):
        return list(available_indices)
    n = len(available_indices)
    if actual_k == 1:
        return [available_indices[n // 2]]
    # Pick actual_k evenly-spaced positions, including endpoints.
    positions = [round(i * (n - 1) / (actual_k - 1)) for i in range(actual_k)]
    return [available_indices[p] for p in positions]


DAM_VIDEO_CAPTION_BODY = (
    "These frames are sampled from a short video clip showing {name}.\n\n"
    "Key historical and cultural facts (use these - do not invent others):\n"
    "{description}\n\n"
    "Notable visual cues to look for:\n{visual_cues}\n\n"
    "Write ONE warm, observant tour-guide paragraph (150 to 300 words, English) that:\n"
    "  - names the landmark explicitly,\n"
    "  - weaves in at least 3 specific facts from the key facts above "
    "(dates, builders, materials, cultural or religious significance, etc.),\n"
    "  - describes what is actually visible across these frames,\n"
    "  - uses warm, observant prose - not a list, not an encyclopedia entry.\n"
    "Do not invent any fact not present above. Do not mention the frames, the camera, "
    "the video, or that you are using a knowledge base or AI."
)


def dam_caption_segment(*, model, frames: list, node: KBNode,
                        full_image_mask_fn, image_token: str) -> str:
    """Call DAM with N frames + N all-ones masks and the KB-grounded prompt.

    `frames` is a list of opaque image objects (PIL.Image at runtime, anything in tests).
    `full_image_mask_fn(frame)` returns the matching all-ones mask for that frame.
    `image_token` is the model's placeholder (DEFAULT_IMAGE_TOKEN at runtime).
    """
    n = len(frames)
    if n == 0:
        raise ValueError("dam_caption_segment requires at least one frame")
    masks = [full_image_mask_fn(f) for f in frames]
    image_tokens = "\n".join([image_token] * n)
    query = (
        image_tokens
        + "\n"
        + DAM_VIDEO_CAPTION_BODY.format(
            name=node.name_en,
            description=node.description_en,
            visual_cues=node.visual_cues_en,
        )
    )
    text = model.get_description(image_pil=frames, mask_pil=masks, query=query)
    return text.strip()


def sample_frames(video_path: Path | str, sample_fps: float) -> list[tuple[int, float, "Image.Image"]]:
    """Read frames from `video_path` at `sample_fps`.

    Returns a list of (frame_idx, timestamp_s, PIL.Image). frame_idx is the
    decoded frame's original position in the source video (useful for debug);
    timestamp_s is the wall-clock time of that frame.

    Returns [] (and logs a warning) for unreadable or empty videos.
    """
    import cv2  # local import keeps the module importable without OpenCV at test-collect time

    path = str(video_path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            log.warning("cv2 could not open video: %s", path)
            return []
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 0:
            log.warning("video has no readable FPS: %s", path)
            return []
        stride = max(1, int(round(src_fps / sample_fps)))
        out: list[tuple[int, float, Image.Image]] = []
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                timestamp_s = frame_idx / src_fps
                out.append((frame_idx, timestamp_s, img))
            frame_idx += 1
        return out
    finally:
        cap.release()


def _full_image_mask(image: "Image.Image") -> "Image.Image":
    arr = np.full((image.size[1], image.size[0]), 255, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _default_retrieve_fn(dino_index_path: str, id_map_path: str):
    """Build a callable that maps a PIL frame -> (kb_id, score). Loads index once."""
    import faiss
    import sys

    # FeatureExtractor lives under scripts/data_collection — add to sys.path once.
    scripts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "data_collection",
    )
    if scripts_dir not in sys.path:
        sys.path.append(scripts_dir)
    from feature_extractor import FeatureExtractor  # type: ignore

    index = faiss.read_index(dino_index_path)
    with open(id_map_path, "r") as f:
        id_map = {int(k): v for k, v in json.load(f).items()}
    extractor = FeatureExtractor()

    def _retrieve(frame_pil):
        feat = extractor.extract_features(frame_pil).astype("float32")
        scores, indices = index.search(feat, k=1)
        idx = int(indices[0][0])
        if idx < 0:
            return None, 0.0
        path = id_map.get(idx)
        if not path:
            return None, float(scores[0][0])
        kb_id = os.path.basename(os.path.dirname(path))
        return kb_id, float(scores[0][0])

    return _retrieve


def _default_dam_caption_fn(model, frames, node):
    from dam import DEFAULT_IMAGE_TOKEN  # type: ignore
    return dam_caption_segment(
        model=model, frames=frames, node=node,
        full_image_mask_fn=_full_image_mask,
        image_token=DEFAULT_IMAGE_TOKEN,
    )


def caption_video(
    video_path: "Path | str",
    kb_nodes: dict,
    dino_index_path: "Path | str",
    id_map_path: "Path | str",
    sample_fps: float = 1.0,
    smooth_window: int = 3,
    min_segment_seconds: float = 2.0,
    dam_frame_budget: "tuple[int, int]" = (4, 8),
    retrieve_fn=None,
    dam_caption_fn=None,
) -> "list[VideoSegment]":
    """See docs/superpowers/specs/2026-05-14-video-caption-pipeline-design.md."""
    # Input validation (raise early, before any model is touched)
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")
    if dam_frame_budget[0] > dam_frame_budget[1] or dam_frame_budget[0] < 1:
        raise ValueError(f"invalid dam_frame_budget: {dam_frame_budget}")
    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)
    if not Path(dino_index_path).exists():
        raise FileNotFoundError(dino_index_path)
    if not Path(id_map_path).exists():
        raise FileNotFoundError(id_map_path)

    # 1. Sample frames
    sampled = sample_frames(video_path, sample_fps=sample_fps)
    if not sampled:
        return []
    stride_s = 1.0 / sample_fps

    # 2. Per-frame retrieval
    if retrieve_fn is None:
        retrieve_fn = _default_retrieve_fn(str(dino_index_path), str(id_map_path))
    nodes_by_kb_id = index_by_kb_id(kb_nodes)
    records: list[FrameRecord] = []
    for _, t, img in sampled:
        kb_id, score = retrieve_fn(img)
        if kb_id is not None and kb_id not in nodes_by_kb_id:
            kb_id = None
        records.append(FrameRecord(timestamp_s=t, kb_id=kb_id, score=score))

    # 3+4. Smooth, group, drop short/unknown
    segs = smooth_and_group(
        records,
        smooth_window=smooth_window,
        min_segment_seconds=min_segment_seconds,
        stride_s=stride_s,
    )
    if not segs:
        return []

    # 5. Per-segment DAM caption
    if dam_caption_fn is None:
        dam_model = registry.get(DAM_NAME)
        dam_caption_fn = lambda frames, node: _default_dam_caption_fn(dam_model, frames, node)

    out: list[VideoSegment] = []
    for s in segs:
        seg_seconds = s["end_s"] - s["start_s"]
        picked = pick_frame_indices(
            segment_seconds=seg_seconds,
            available_indices=s["frame_indices"],
            budget=dam_frame_budget,
        )
        picked_frames = [sampled[p][2] for p in picked]
        node = nodes_by_kb_id[s["kb_id"]]
        t0 = time.perf_counter()
        try:
            caption = dam_caption_fn(picked_frames, node)
        except Exception:
            log.warning("DAM caption failed for segment %.2f-%.2f (%s); dropping",
                        s["start_s"], s["end_s"], s["kb_id"], exc_info=True)
            continue
        dam_seconds = time.perf_counter() - t0
        out.append(VideoSegment(
            start_s=s["start_s"],
            end_s=s["end_s"],
            kb_id=s["kb_id"],
            node_id=node.id,
            name_en=node.name_en,
            confidence=s["confidence"],
            caption=caption,
            debug={
                "frames_total": len(s["frame_indices"]),
                "frames_sampled": len(picked_frames),
                "timings": {"dam_caption": dam_seconds},
            },
        ))
    return out
