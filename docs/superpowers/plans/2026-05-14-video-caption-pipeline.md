# Video Caption Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `hanoi_caption.video_pipeline.caption_video(video_path, ...)` that turns a local video file into a list of timestamped, landmark-grounded `VideoSegment` records — reusing the existing DINOv3 FAISS index and DAM-3B model.

**Architecture:** New module `hanoi_caption/video_pipeline.py`. Five pure helpers (frame sampler, retriever, smoother, frame-budget calculator, DAM multi-frame caller) wired by one entry function `caption_video`. The four pure ones get unit-tested with synthetic data; the slow integration is covered by one GPU-only smoke test using a programmatically-built fixture video.

**Tech Stack:** Python 3.11, OpenCV (`cv2`) for video I/O, FAISS for retrieval, NVIDIA `dam` package's multi-image `get_description` path, Pydantic v2 schemas, pytest.

**Spec:** `docs/superpowers/specs/2026-05-14-video-caption-pipeline-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `hanoi_caption/schemas.py` | Modify | Add `VideoSegment` Pydantic model |
| `hanoi_caption/video_pipeline.py` | Create | Sampling, retrieval, smoothing, budget calc, DAM call, `caption_video` entry |
| `tests/test_video_smoothing.py` | Create | Unit tests for grouping + min-duration filter |
| `tests/test_video_frame_budget.py` | Create | Unit tests for adaptive K calculation |
| `tests/test_video_dam_caption.py` | Create | Unit tests for multi-frame prompt builder (mock DAM) |
| `tests/test_caption_video_smoke.py` | Create | One slow GPU smoke test on a synthesised fixture clip |
| `tests/fixtures/video/` | Create dir | Holds the synthesised `.mp4` fixtures |
| `tests/conftest.py` | Modify | Add `slow` marker registration + `fixture_video` fixture |

---

## Task 1: Add `VideoSegment` schema

**Files:**
- Modify: `hanoi_caption/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1.1: Write the failing test**

Add to `tests/test_schemas.py` (append, do not replace existing tests):

```python
def test_video_segment_round_trip():
    from hanoi_caption.schemas import VideoSegment

    seg = VideoSegment(
        start_s=0.0,
        end_s=4.0,
        kb_id="temple_of_literature",
        node_id="69cfe5ab0a741c71017316fd",
        name_en="Temple of Literature",
        confidence=0.83,
        caption="A grand pavilion...",
    )
    assert seg.start_s == 0.0
    assert seg.end_s == 4.0
    assert seg.kb_id == "temple_of_literature"
    assert seg.debug == {}


def test_video_segment_accepts_debug_payload():
    from hanoi_caption.schemas import VideoSegment

    seg = VideoSegment(
        start_s=0.0, end_s=2.0,
        kb_id="x", node_id="y", name_en="X",
        confidence=0.5, caption="...",
        debug={"frames_sampled": 4, "timings": {"dam_caption": 12.3}},
    )
    assert seg.debug["frames_sampled"] == 4
    assert seg.debug["timings"]["dam_caption"] == 12.3
```

- [ ] **Step 1.2: Run test to verify it fails**

```
cd D:/Jupiter/captioning-system
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_schemas.py -v -k video_segment
```

Expected: `ImportError` / `AttributeError` — `VideoSegment` does not exist yet.

- [ ] **Step 1.3: Add the schema**

Append to `hanoi_caption/schemas.py`, after the existing `CaptionResult` class:

```python
class VideoSegment(BaseModel):
    start_s: float
    end_s: float
    kb_id: str
    node_id: str
    name_en: str
    confidence: float
    caption: str
    debug: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 1.4: Run test to verify it passes**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_schemas.py -v -k video_segment
```

Expected: 2 passed.

- [ ] **Step 1.5: Commit**

```
git add hanoi_caption/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add VideoSegment for video pipeline output"
```

---

## Task 2: Smoothing + grouping (pure function)

**Files:**
- Create: `hanoi_caption/video_pipeline.py`
- Test: `tests/test_video_smoothing.py`

The smoother takes a list of per-frame retrieval records and returns a list of merged segments. Pure Python — no models, no I/O. This is the most algorithmically tricky piece; tests should cover boundary cases explicitly.

Per-frame input record: a `dataclass` of `(timestamp_s: float, kb_id: str | None, score: float)`. `kb_id is None` means "unknown" (no confident landmark for this frame).

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_video_smoothing.py`:

```python
from hanoi_caption.video_pipeline import FrameRecord, smooth_and_group


def _records(seq, start=0.0, stride=1.0):
    """Build FrameRecord list from a string like 'AAABBB' or list of kb_ids."""
    if isinstance(seq, str):
        seq = list(seq)
    return [
        FrameRecord(timestamp_s=start + i * stride,
                    kb_id=(None if c in ("?", None) else c),
                    score=0.9)
        for i, c in enumerate(seq)
    ]


def test_groups_consecutive_same_kb_id():
    segs = smooth_and_group(_records("AAAABBBB"), smooth_window=1, min_segment_seconds=0.0, stride_s=1.0)
    kb_ids = [s["kb_id"] for s in segs]
    assert kb_ids == ["A", "B"]
    assert segs[0]["start_s"] == 0.0 and segs[0]["end_s"] == 4.0
    assert segs[1]["start_s"] == 4.0 and segs[1]["end_s"] == 8.0


def test_majority_vote_absorbs_single_flicker():
    # ABA at the start of a B run -> middle A is a flicker; window=3 majority is B
    segs = smooth_and_group(_records("BBBABBB"), smooth_window=3, min_segment_seconds=0.0, stride_s=1.0)
    assert [s["kb_id"] for s in segs] == ["B"]


def test_unknown_runs_are_dropped():
    segs = smooth_and_group(_records("AAA???BBB"), smooth_window=1, min_segment_seconds=0.0, stride_s=1.0)
    assert [s["kb_id"] for s in segs] == ["A", "B"]


def test_short_run_absorbed_into_longer_neighbor():
    # 4s A, 1s B, 4s A -> the 1s B gets absorbed; result is one A segment
    segs = smooth_and_group(_records("AAAABAAAA"), smooth_window=1, min_segment_seconds=2.0, stride_s=1.0)
    assert len(segs) == 1
    assert segs[0]["kb_id"] == "A"
    assert segs[0]["start_s"] == 0.0
    assert segs[0]["end_s"] == 9.0


def test_short_run_at_video_start_absorbed_into_only_neighbor():
    segs = smooth_and_group(_records("ABBBBBBB"), smooth_window=1, min_segment_seconds=2.0, stride_s=1.0)
    # leading A is shorter than 2s and has only one neighbor (B) -> absorbed
    assert [s["kb_id"] for s in segs] == ["B"]


def test_short_run_between_unknowns_is_dropped():
    segs = smooth_and_group(_records("???A???"), smooth_window=1, min_segment_seconds=2.0, stride_s=1.0)
    assert segs == []


def test_empty_input_returns_empty():
    assert smooth_and_group([], smooth_window=3, min_segment_seconds=2.0, stride_s=1.0) == []


def test_all_unknown_returns_empty():
    segs = smooth_and_group(_records("?????"), smooth_window=3, min_segment_seconds=2.0, stride_s=1.0)
    assert segs == []


def test_confidence_is_mean_over_segment_frames():
    from hanoi_caption.video_pipeline import FrameRecord
    recs = [
        FrameRecord(0.0, "A", 0.8),
        FrameRecord(1.0, "A", 0.6),
        FrameRecord(2.0, "A", 1.0),
    ]
    segs = smooth_and_group(recs, smooth_window=1, min_segment_seconds=0.0, stride_s=1.0)
    assert len(segs) == 1
    assert abs(segs[0]["confidence"] - 0.8) < 1e-9


def test_segment_carries_frame_indices_for_downstream_sampling():
    segs = smooth_and_group(_records("AAA"), smooth_window=1, min_segment_seconds=0.0, stride_s=1.0)
    assert segs[0]["frame_indices"] == [0, 1, 2]
```

- [ ] **Step 2.2: Run tests to verify they fail**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_video_smoothing.py -v
```

Expected: `ModuleNotFoundError: hanoi_caption.video_pipeline`.

- [ ] **Step 2.3: Implement the smoother**

Create `hanoi_caption/video_pipeline.py`:

```python
"""Video captioning pipeline: DINOv3 retrieval per frame + DAM multi-frame caption per segment.

See `docs/superpowers/specs/2026-05-14-video-caption-pipeline-design.md`.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


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


def _absorb_short_runs(runs: list[dict[str, Any]], min_seconds: float) -> list[dict[str, Any]]:
    """Absorb short runs into the longer neighbor.

    Rules:
    - Unknown (kb_id is None) runs are NEVER absorbed into a known neighbor; they are
      simply dropped at the final filter step. This function only merges short *known*
      runs into adjacent known runs of the same kb_id, OR drops them when surrounded
      only by unknowns / video boundaries with no known neighbor.
    - Among two known neighbors, absorb into the longer (in duration). Ties go to the
      preceding neighbor.
    """
    if not runs:
        return runs

    # Repeatedly find a short known run and try to absorb it. Stop when no more changes.
    changed = True
    while changed:
        changed = False
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
            # Absorb: relabel this run to target's kb_id; recompute via second pass after loop.
            runs[i] = {**run, "kb_id": target["kb_id"]}
            changed = True
            break

    # Second pass: merge now-adjacent runs sharing kb_id.
    merged: list[dict[str, Any]] = []
    for run in runs:
        if merged and merged[-1]["kb_id"] == run["kb_id"]:
            prev = merged[-1]
            new_indices = prev["frame_indices"] + run["frame_indices"]
            new_scores = [
                run["confidence"] * len(run["frame_indices"]),
                prev["confidence"] * len(prev["frame_indices"]),
            ]
            n_total = len(new_indices)
            merged[-1] = {
                "kb_id": prev["kb_id"],
                "start_s": prev["start_s"],
                "end_s": run["end_s"],
                "frame_indices": new_indices,
                "confidence": sum(new_scores) / n_total,
            }
        else:
            merged.append(run)
    return merged


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
    runs = _absorb_short_runs(runs, min_segment_seconds)
    return [r for r in runs if r["kb_id"] is not None]
```

- [ ] **Step 2.4: Run tests to verify they pass**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_video_smoothing.py -v
```

Expected: 10 passed.

- [ ] **Step 2.5: Commit**

```
git add hanoi_caption/video_pipeline.py tests/test_video_smoothing.py
git commit -m "feat(video): smoothing + segment grouping pure helper"
```

---

## Task 3: Adaptive frame-budget calculator (pure function)

**Files:**
- Modify: `hanoi_caption/video_pipeline.py`
- Test: `tests/test_video_frame_budget.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/test_video_frame_budget.py`:

```python
from hanoi_caption.video_pipeline import pick_frame_indices


def test_target_k_is_segment_seconds_clamped_to_budget():
    # 3s segment, plenty of frames available -> target K = max(min=4, 3) = 4
    idx = pick_frame_indices(segment_seconds=3.0, available_indices=list(range(10)), budget=(4, 8))
    assert len(idx) == 4


def test_long_segment_caps_at_max_budget():
    # 20s segment, lots of frames -> capped at max=8
    idx = pick_frame_indices(segment_seconds=20.0, available_indices=list(range(40)), budget=(4, 8))
    assert len(idx) == 8


def test_no_upsampling_when_fewer_frames_than_min():
    # only 3 frames available; budget min is 4 -> return all 3 (no padding/duplication)
    idx = pick_frame_indices(segment_seconds=5.0, available_indices=[10, 11, 12], budget=(4, 8))
    assert idx == [10, 11, 12]


def test_indices_are_subset_of_available_and_sorted():
    avail = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    idx = pick_frame_indices(segment_seconds=10.0, available_indices=avail, budget=(4, 8))
    assert all(i in avail for i in idx)
    assert idx == sorted(idx)


def test_indices_are_evenly_spaced():
    # 8 picks from 10 available -> roughly even spacing; first and last included
    avail = list(range(10))
    idx = pick_frame_indices(segment_seconds=20.0, available_indices=avail, budget=(4, 8))
    assert idx[0] == 0
    assert idx[-1] == 9
    assert len(idx) == 8


def test_empty_available_returns_empty():
    assert pick_frame_indices(segment_seconds=5.0, available_indices=[], budget=(4, 8)) == []


def test_invalid_budget_raises():
    import pytest
    with pytest.raises(ValueError):
        pick_frame_indices(segment_seconds=5.0, available_indices=[1, 2, 3], budget=(8, 4))
```

- [ ] **Step 3.2: Run tests to verify they fail**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_video_frame_budget.py -v
```

Expected: `ImportError` / `AttributeError`.

- [ ] **Step 3.3: Implement the budget picker**

Append to `hanoi_caption/video_pipeline.py`:

```python
import math


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
    # Pick actual_k evenly-spaced positions, including endpoints.
    positions = [round(i * (n - 1) / (actual_k - 1)) for i in range(actual_k)]
    return [available_indices[p] for p in positions]
```

- [ ] **Step 3.4: Run tests to verify they pass**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_video_frame_budget.py -v
```

Expected: 7 passed.

- [ ] **Step 3.5: Commit**

```
git add hanoi_caption/video_pipeline.py tests/test_video_frame_budget.py
git commit -m "feat(video): adaptive frame-budget picker"
```

---

## Task 4: Multi-frame DAM caption helper

**Files:**
- Modify: `hanoi_caption/video_pipeline.py`
- Test: `tests/test_video_dam_caption.py`

DAM-3B's multi-image API requires the prompt to contain one `<image>` placeholder per frame. The model package exposes `DEFAULT_IMAGE_TOKEN` for that placeholder.

This helper is testable in isolation by injecting a mock `dam_model` and a mock `full_image_mask_fn` (so the test does not need PIL machinery).

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_video_dam_caption.py`:

```python
from hanoi_caption.schemas import KBNode
from hanoi_caption.video_pipeline import dam_caption_segment


class _MockModel:
    def __init__(self):
        self.last_call = None

    def get_description(self, image_pil, mask_pil, query, **kwargs):
        # DAM's signature: lists for multi-image input
        self.last_call = {
            "image_pil": image_pil,
            "mask_pil": mask_pil,
            "query": query,
            "kwargs": kwargs,
        }
        return "  a caption  "


def _node():
    return KBNode(
        id="x", kb_id="temple_of_literature",
        name_en="Temple of Literature", name_vi="t",
        type="object", parent_id=None,
        description_en="A famous temple.", description_vi="",
        visual_cues_en="red roof, wooden columns", visual_cues_vi="",
        tags=[],
    )


def test_prompt_includes_one_image_token_per_frame():
    model = _MockModel()
    frames = ["frame1", "frame2", "frame3"]  # opaque to the function

    dam_caption_segment(
        model=model,
        frames=frames,
        node=_node(),
        full_image_mask_fn=lambda f: f"mask_of_{f}",
        image_token="<image>",
    )

    # 3 frames -> 3 <image> tokens in the assembled prompt
    assert model.last_call["query"].count("<image>") == 3


def test_prompt_includes_landmark_name_and_kb_facts():
    model = _MockModel()
    dam_caption_segment(
        model=model,
        frames=["f"],
        node=_node(),
        full_image_mask_fn=lambda f: "m",
        image_token="<image>",
    )
    q = model.last_call["query"]
    assert "Temple of Literature" in q
    assert "A famous temple." in q
    assert "red roof, wooden columns" in q


def test_passes_frames_and_masks_as_parallel_lists():
    model = _MockModel()
    frames = ["a", "b", "c"]
    dam_caption_segment(
        model=model, frames=frames, node=_node(),
        full_image_mask_fn=lambda f: f"M:{f}",
        image_token="<image>",
    )
    assert model.last_call["image_pil"] == ["a", "b", "c"]
    assert model.last_call["mask_pil"] == ["M:a", "M:b", "M:c"]


def test_returns_stripped_caption():
    model = _MockModel()
    out = dam_caption_segment(
        model=model, frames=["f"], node=_node(),
        full_image_mask_fn=lambda f: "m",
        image_token="<image>",
    )
    assert out == "a caption"
```

- [ ] **Step 4.2: Run tests to verify they fail**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_video_dam_caption.py -v
```

Expected: `ImportError` / `AttributeError`.

- [ ] **Step 4.3: Implement the multi-frame caption helper**

Append to `hanoi_caption/video_pipeline.py`:

```python
from hanoi_caption.schemas import KBNode

DAM_VIDEO_CAPTION_BODY = (
    "These frames are sampled from a short video clip showing {name}.\n\n"
    "Historical and cultural context:\n{description}\n\n"
    "Notable visual features that may be present: {visual_cues}\n\n"
    "Write ONE warm, observant tour-guide paragraph (150 to 300 words, English) "
    "describing what is visible across these frames and weaving in the historical "
    "context above. Do not invent facts beyond what is provided. Write prose, "
    "not a list. Do not mention the frames, the camera, the video, or that you "
    "are using a knowledge base or AI."
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
```

- [ ] **Step 4.4: Run tests to verify they pass**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_video_dam_caption.py -v
```

Expected: 4 passed.

- [ ] **Step 4.5: Commit**

```
git add hanoi_caption/video_pipeline.py tests/test_video_dam_caption.py
git commit -m "feat(video): multi-frame DAM caption prompt builder"
```

---

## Task 5: Frame sampler (cv2-based)

**Files:**
- Modify: `hanoi_caption/video_pipeline.py`

The sampler opens a video with cv2, computes the strided index range from the video's FPS, reads matching frames, converts BGR→RGB→PIL. No dedicated unit test — the integration smoke test (Task 7) exercises this path. Keep the function small so reading it is the verification.

- [ ] **Step 5.1: Implement the sampler**

Append to `hanoi_caption/video_pipeline.py`:

```python
from pathlib import Path
from PIL import Image
import logging

log = logging.getLogger(__name__)


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
```

- [ ] **Step 5.2: Verify the module still imports cleanly**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -c "from hanoi_caption.video_pipeline import sample_frames, smooth_and_group, pick_frame_indices, dam_caption_segment; print('ok')"
```

Expected: `ok`

- [ ] **Step 5.3: Commit**

```
git add hanoi_caption/video_pipeline.py
git commit -m "feat(video): cv2-based frame sampler"
```

---

## Task 6: Build a fixture video for the smoke test

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/fixtures/video/` (directory)

The smoke test in Task 7 needs an actual mp4. Generate one in a session-scoped fixture that builds a 12-second clip from existing image fixtures: 4s of `temple_of_literature_1.jpg`, 4s of `hoangthanh.jpg`, 4s of `gahanoi.jpg` at 5 fps (60 total frames).

- [ ] **Step 6.1: Add the marker registration and fixture**

Replace `tests/conftest.py` with:

```python
"""Shared pytest fixtures."""
from pathlib import Path
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that load real models or hit a GPU")


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory) -> Path:
    """Build a deterministic 12s test video from three existing image fixtures.

    Layout (5 fps, 60 frames total):
        frames  0..19  -> temple_of_literature_1.jpg
        frames 20..39  -> hoangthanh.jpg
        frames 40..59  -> gahanoi.jpg
    """
    import cv2
    import numpy as np
    from PIL import Image

    src_dir = Path(__file__).parent / "fixtures"
    triplet = [
        src_dir / "temple_of_literature_1.jpg",
        src_dir / "hoangthanh.jpg",
        src_dir / "gahanoi.jpg",
    ]
    for p in triplet:
        if not p.exists():
            pytest.skip(f"required fixture missing: {p}")

    out_dir = tmp_path_factory.mktemp("video")
    out_path = out_dir / "synthetic_tour.mp4"

    # Normalise all source images to a common size (smallest common WxH).
    pil_imgs = [Image.open(p).convert("RGB") for p in triplet]
    w = min(im.size[0] for im in pil_imgs)
    h = min(im.size[1] for im in pil_imgs)
    w -= w % 2
    h -= h % 2  # cv2 mp4 writer needs even dimensions
    resized = [im.resize((w, h), Image.LANCZOS) for im in pil_imgs]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 5
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for im in resized:
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        for _ in range(fps * 4):  # 4 seconds per image
            writer.write(bgr)
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0
    return out_path


@pytest.fixture(scope="session")
def kb_nodes_real():
    """Load the real KB. Skipped if not vendored."""
    kb_path = Path("data/kb.json")
    if not kb_path.exists():
        pytest.skip(f"data/kb.json not present")
    from hanoi_caption.kb_loader import load_kb
    return load_kb(kb_path, only_objects=True)
```

- [ ] **Step 6.2: Smoke-test the fixture itself**

Create a one-off check (do NOT commit this — it's just to confirm fixture writes a valid file):

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/conftest.py --collect-only -q
```

Expected: `0 items collected` (conftest contains no tests). No errors.

Then verify by running an existing test to confirm the conftest still loads:

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_kb_loader.py -v
```

Expected: existing tests still pass.

- [ ] **Step 6.3: Commit**

```
git add tests/conftest.py
git commit -m "test(video): add synthetic-video fixture and slow marker"
```

---

## Task 7: Wire `caption_video` + smoke test

**Files:**
- Modify: `hanoi_caption/video_pipeline.py`
- Create: `tests/test_caption_video_smoke.py`

The top-level function loads the FAISS index + id map, runs the four helpers in sequence, and emits `VideoSegment` records. Per the spec, the heavy bits (DINOv3 retrieval and DAM call) accept dependency-injection hooks so unit tests can mock them. The smoke test exercises the full real pipeline against the synthetic video.

- [ ] **Step 7.1: Write the failing smoke test**

Create `tests/test_caption_video_smoke.py`:

```python
import pytest


@pytest.mark.slow
def test_caption_video_smoke(fixture_video, kb_nodes_real):
    """End-to-end on the synthetic 12s clip. GPU + DAM + DINOv3 required."""
    from hanoi_caption.video_pipeline import caption_video

    segments = caption_video(
        video_path=fixture_video,
        kb_nodes=kb_nodes_real,
        dino_index_path="data/cache/dino_faiss.index",
        id_map_path="data/cache/id_map.json",
        sample_fps=1.0,
        smooth_window=3,
        min_segment_seconds=2.0,
        dam_frame_budget=(4, 8),
    )

    assert isinstance(segments, list)
    assert len(segments) >= 1, "expected at least one landmark segment on the synthetic clip"

    # Each segment is well-formed and within video bounds.
    for seg in segments:
        assert seg.start_s >= 0.0
        assert seg.end_s > seg.start_s
        assert seg.end_s <= 12.5  # 12s clip + small slack
        assert seg.caption.strip()
        assert seg.kb_id and seg.node_id and seg.name_en

    # Segments are sorted, non-overlapping.
    starts = [s.start_s for s in segments]
    assert starts == sorted(starts)
    for a, b in zip(segments, segments[1:]):
        assert a.end_s <= b.start_s + 1e-6
```

Also add light unit tests that exercise the validation path without needing the GPU. Add to the same file:

```python
def test_caption_video_validates_sample_fps(tmp_path):
    from hanoi_caption.video_pipeline import caption_video
    v = tmp_path / "empty.mp4"
    v.write_bytes(b"")  # non-existent check is BEFORE we look at contents; file exists so we reach validation
    with pytest.raises(ValueError, match="sample_fps"):
        caption_video(
            video_path=v, kb_nodes={},
            dino_index_path="x", id_map_path="x",
            sample_fps=0.0,
        )


def test_caption_video_validates_frame_budget(tmp_path):
    from hanoi_caption.video_pipeline import caption_video
    v = tmp_path / "empty.mp4"
    v.write_bytes(b"")
    with pytest.raises(ValueError, match="dam_frame_budget"):
        caption_video(
            video_path=v, kb_nodes={},
            dino_index_path="x", id_map_path="x",
            dam_frame_budget=(8, 4),
        )


def test_caption_video_missing_file_raises(tmp_path):
    from hanoi_caption.video_pipeline import caption_video
    with pytest.raises(FileNotFoundError):
        caption_video(
            video_path=tmp_path / "nope.mp4", kb_nodes={},
            dino_index_path="x", id_map_path="x",
        )
```

- [ ] **Step 7.2: Run validation tests to verify they fail**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_caption_video_smoke.py -v -m "not slow"
```

Expected: 3 failures — `caption_video` does not exist yet.

- [ ] **Step 7.3: Implement `caption_video`**

Append to `hanoi_caption/video_pipeline.py`:

```python
import json
import time
import os

import numpy as np

from hanoi_caption.kb_loader import index_by_kb_id
from hanoi_caption.model_registry import registry
from hanoi_caption.region_describer import MODEL_NAME as DAM_NAME
from hanoi_caption.schemas import VideoSegment


def _full_image_mask(image: "Image.Image") -> "Image.Image":
    arr = np.full((image.size[1], image.size[0]), 255, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _default_retrieve_fn(dino_index_path: str, id_map_path: str):
    """Build a callable that maps a PIL frame -> (kb_id, score). Loads index once."""
    import faiss
    sys_path_inserted = False
    try:
        # FeatureExtractor lives under scripts/data_collection — add to sys.path once.
        import sys
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "scripts", "data_collection")
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)
            sys_path_inserted = True
        from feature_extractor import FeatureExtractor  # type: ignore
    finally:
        pass

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
    video_path: Path | str,
    kb_nodes: dict[str, KBNode],
    dino_index_path: Path | str,
    id_map_path: Path | str,
    sample_fps: float = 1.0,
    smooth_window: int = 3,
    min_segment_seconds: float = 2.0,
    dam_frame_budget: tuple[int, int] = (4, 8),
    retrieve_fn=None,
    dam_caption_fn=None,
) -> list[VideoSegment]:
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
        # Unknown if FAISS returned a folder slug we don't have in the loaded KB
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

    frames_by_idx = {idx: img for idx, _, img in sampled}
    out: list[VideoSegment] = []
    for s in segs:
        seg_seconds = s["end_s"] - s["start_s"]
        picked = pick_frame_indices(
            segment_seconds=seg_seconds,
            available_indices=s["frame_indices"],
            budget=dam_frame_budget,
        )
        # frame_indices in `s` are positions in the `sampled` list, not raw video frame ids
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
```

- [ ] **Step 7.4: Run the fast tests to verify they pass**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_caption_video_smoke.py -v -m "not slow"
```

Expected: 3 passed (validation tests). The `@pytest.mark.slow` one is skipped.

- [ ] **Step 7.5: Run the slow smoke test end-to-end**

This loads DAM-3B and DINOv3; runs against the synthetic 12s clip; takes ~1-2 minutes:

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/test_caption_video_smoke.py -v -m "slow"
```

Expected: 1 passed. At least one `VideoSegment` returned with a non-empty caption.

If it fails:
- `FileNotFoundError: data/cache/dino_faiss.index` → ensure you ran `python scripts/data_collection/indexer.py data/kb_images` and the index file is present.
- DAM CUDA assert / OOM → free GPU memory (close other notebooks/processes), retry.
- Zero segments returned → check the synthetic video's matched landmarks via the existing notebook cell first; the smoke clip mixes temple_of_literature_1 / hoangthanh / gahanoi and at least one should clear the smoothing filter.

- [ ] **Step 7.6: Run the whole test suite to confirm nothing else broke**

```
"D:/Jupiter/luna_env/Scripts/python.exe" -m pytest tests/ -v -m "not slow"
```

Expected: all pre-existing fast tests still pass; the new ones pass.

- [ ] **Step 7.7: Commit**

```
git add hanoi_caption/video_pipeline.py tests/test_caption_video_smoke.py
git commit -m "feat(video): caption_video entry function + smoke test"
```

---

## Done

After Task 7, `caption_video(...)` is callable from notebooks and scripts. The pipeline reuses the existing FAISS index, KB, and DAM registry — no new model downloads. The next sensible follow-up (out of scope for this plan): a notebook cell that calls `caption_video` on a real Hanoi walking-tour clip and renders the segments with timestamps.
