# Video Caption Pipeline — Design

**Date:** 2026-05-14
**Status:** Approved (pending spec review)

## Goal

Extend the existing image captioning system to accept a local video file and emit a list of timestamped, landmark-grounded captions — one per detected landmark segment. Reuse the existing DINOv3 retrieval index, KB loader, and DAM-3B model. No new heavyweight models; no video-native VLM.

## Background

The current `dinov3_rag_cell` notebook pipeline takes a single PIL image, runs DINOv3 against the FAISS index built from `data/kb_images/`, looks up the matched landmark by `kb_id`, and asks DAM-3B for a KB-grounded caption. Latency budget per image is roughly 10–15 s (DAM dominates; DINOv3 retrieval is ~25 ms).

For video, the user wants segment-by-segment captions with timestamps. The same retrieve-then-describe flow applies per segment, with two new concerns: (1) deciding where segment boundaries fall, and (2) handing DAM more than one frame so the description reflects motion / multi-view context rather than a single static keyframe.

## Architecture & Data Flow

```
video file (local path)
   │
   ▼
[1] Frame sampler — cv2 reads the video, emits frames at sample_fps (default 1.0 Hz)
   │   yields (frame_idx, timestamp_s, PIL.Image)
   ▼
[2] Per-frame retrieval — FeatureExtractor (DINOv3) → FAISS top-1 → kb_id
   │   per-frame record: (timestamp_s, kb_id, score)
   ▼
[3] Smoothing — 3-frame sliding-window majority vote over the kb_id stream,
   │   then merge consecutive frames with the same kb_id into runs
   │   list[(start_s, end_s, kb_id, frame_indices)]
   ▼
[4] Min-duration filter — any run shorter than min_segment_seconds (default 2.0)
   │   is absorbed into the longer of its two neighbors (ties broken by
   │   absorbing into the preceding neighbor). A short run at the very start
   │   or end of the video has only one neighbor and is absorbed into it.
   │   A short run whose only neighbors are "unknown" is dropped.
   │   Unknown / low-confidence runs are also dropped
   │   (per the "skip non-landmark spans" decision).
   ▼
[5] Per-segment caption — for each surviving segment:
   │   - target_K = clamp(ceil(segment_seconds), min=dam_frame_budget[0],
   │                                              max=dam_frame_budget[1])
   │   - actual_K = min(target_K, len(segment_frames))   (no upsampling)
   │   - pick actual_K evenly-spaced frames from the segment
   │   - build prompt = (DEFAULT_IMAGE_TOKEN + "\n") * actual_K + DAM_CAPTION_PROMPT_BODY
   │     (so the model receives one <image> placeholder per frame)
   │   - call model.get_description(image_pil=[f1..fK], mask_pil=[m1..mK], query=prompt)
   │     where each mask is a full-image all-ones mask (matching pipeline_retriever)
   ▼
list[VideoSegment]
```

The standard `dam_3b` model already supports multi-image input via this pathway — `get_description_from_prompt_iterator` builds one image tensor per (image, mask) pair and passes them all into a single `model.generate(input_ids=..., images=[t1..tK])` call. The `<image>` tokens in `prompt` are replaced by each frame's visual features. No `nvidia/DAM-3B-Video` checkpoint is needed.

## Module Layout

New file: `hanoi_caption/video_pipeline.py`. Mirrors the structure of `pipeline_retriever.py`: stage-timed, helper functions for each stage, a single public entry function.

New schema in `hanoi_caption/schemas.py`: `VideoSegment` (see below).

No new model registry entries. No new dependencies — `cv2` is already pulled in via `scripts/data_collection/feature_extractor.py`'s install path; `numpy`, `PIL`, `faiss`, and `dam` are already used.

## Public API

```python
# hanoi_caption/video_pipeline.py
def caption_video(
    video_path: Path | str,
    kb_nodes: dict[str, KBNode],
    dino_index_path: Path | str = "data/cache/dino_faiss.index",
    id_map_path:     Path | str = "data/cache/id_map.json",
    sample_fps:          float = 1.0,
    smooth_window:       int   = 3,
    min_segment_seconds: float = 2.0,
    dam_frame_budget:    tuple[int, int] = (4, 8),
) -> list[VideoSegment]: ...
```

Behavior contract:
- Raises `FileNotFoundError` if `video_path` does not exist (before any model is touched).
- Raises `FileNotFoundError` if the DINOv3 index or id map cannot be opened.
- Returns an empty list (no exception) for: an unreadable video, a readable but empty video, or a video where every segment was filtered out.
- DAM failure on a single segment is caught, logged via `logging.warning`, and the segment is omitted; processing continues with the remaining segments.

## Schemas

```python
# hanoi_caption/schemas.py — appended

class VideoSegment(BaseModel):
    start_s: float
    end_s:   float
    kb_id:   str            # human-readable slug used in folder paths
    node_id: str            # opaque KBNode.id
    name_en: str            # display name from KBNode
    confidence: float       # mean DINOv3 top-1 score across the segment's frames
    caption: str            # DAM-3B caption for this segment
    debug: dict[str, Any] = Field(default_factory=dict)
    # debug keys reserved: frames_total, frames_sampled, timings={dam_caption: float}
```

## Error & Edge Cases

| Case | Behavior |
|---|---|
| `video_path` missing | `FileNotFoundError` before model load |
| `video_path` exists but cv2 cannot open it | Empty list, warning logged |
| Video has zero readable frames | Empty list, warning logged |
| All segments are below `min_segment_seconds` | Empty list (silent — this is a valid "no landmark" outcome) |
| FAISS path lookup returns a `kb_id` not in `kb_nodes` | Frame contributes an "unknown" vote, segment is treated as non-landmark and skipped |
| DAM generation raises on one segment | Segment dropped, `logging.warning(...)`, other segments continue |
| `dam_frame_budget` reversed (max < min) | `ValueError` at function entry |
| `sample_fps <= 0` | `ValueError` at function entry |
| Segment has fewer frames than `dam_frame_budget[0]` | All available frames are used (no upsampling, no padding) |

## Testing

Unit tests (pure Python, no model, fast):

- `test_smoothing.py` — synthetic kb_id sequences exercise:
  - 3-frame majority vote absorbs a 1-frame flicker
  - Runs are merged by consecutive identity
  - Short runs (under `min_segment_seconds`) are absorbed into the longer neighbor
  - "Unknown" runs are dropped, not absorbed
  - Boundary cases: single-frame video, all-unknown video, all-same-landmark video
- `test_segment_frame_budget.py` — subsampling within a segment:
  - K = min when frames available ≤ min
  - K = max when many frames available
  - Evenly-spaced indices

Integration smoke test (slow, GPU-required, marked `@pytest.mark.slow`):

- `test_caption_video_smoke.py` — runs `caption_video` on a curated short fixture clip under `tests/fixtures/video/`. Asserts:
  - Result is a non-empty list
  - At least one segment matches a known landmark in the clip
  - All segment `start_s < end_s`, no overlap, total span ≤ video duration

No snapshot tests of caption text — DAM output is non-deterministic.

## Out of Scope

- Streaming / realtime input (live camera, RTSP). Local file only.
- Audio extraction or speech transcription.
- Output formatting to SRT/VTT — emitted segments contain enough info for a thin formatter to live in a separate module later.
- Detecting multiple landmarks within a single frame (the pipeline groups by top-1 only).
- Scene-change detection via pixel diff / PySceneDetect. Landmark-change is the only segmentation signal.
- Notebook integration. A demo cell can be added once the module passes tests; that's an executing-plans concern, not a design one.

## Open Questions

None at this time.
