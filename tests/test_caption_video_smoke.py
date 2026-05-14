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

    for seg in segments:
        assert seg.start_s >= 0.0
        assert seg.end_s > seg.start_s
        assert seg.end_s <= 12.5
        assert seg.caption.strip()
        assert seg.kb_id and seg.node_id and seg.name_en

    starts = [s.start_s for s in segments]
    assert starts == sorted(starts)
    for a, b in zip(segments, segments[1:]):
        assert a.end_s <= b.start_s + 1e-6


def test_caption_video_validates_sample_fps(tmp_path):
    from hanoi_caption.video_pipeline import caption_video
    v = tmp_path / "empty.mp4"
    v.write_bytes(b"")
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
