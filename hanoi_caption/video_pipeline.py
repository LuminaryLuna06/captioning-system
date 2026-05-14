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
