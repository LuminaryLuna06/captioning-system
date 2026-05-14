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
