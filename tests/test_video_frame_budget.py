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
