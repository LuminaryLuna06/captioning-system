import json
import pytest
from pathlib import Path

MINI_DATASET = {
    "videos": [
        {
            "id": "vid1", "filename": "A.MOV", "duration": 30.0, "fps": 30, "height": 1080, "width": 1920,
            "segments": [{
                "id": "s1", "name": "Pen Tower", "start_time": 0.0, "end_time": 30.0, "duration": 30.0,
                "regions": [{
                    "captions": {"en": {"combined": "The Pen Tower stands tall.", "visual": "A tall stone tower."}},
                    "knowledge_base_ids": ["kb_node_001"],
                    "knowledge_base_items": [{"description": "Five-story stone tower built in 1865."}]
                }],
                "segment_captions": []
            }]
        },
        {
            "id": "vid2", "filename": "B.MOV", "duration": 20.0, "fps": 30, "height": 1080, "width": 1920,
            "segments": [{
                "id": "s2", "name": "Banh mi", "start_time": 0.0, "end_time": 20.0, "duration": 20.0,
                "regions": [{
                    "captions": {"en": {"combined": "A banh mi sandwich.", "visual": "Bread and fillings."}},
                    "knowledge_base_ids": [],
                    "knowledge_base_items": []
                }],
                "segment_captions": []
            }]
        },
        {
            "id": "vid3", "filename": "C.MOV", "duration": 25.0, "fps": 30, "height": 1080, "width": 1920,
            "segments": [{
                "id": "s3", "name": "Hoan Kiem Lake", "start_time": 0.0, "end_time": 25.0, "duration": 25.0,
                "regions": [{
                    "captions": {"en": {"combined": "The serene lake.", "visual": "A body of water."}},
                    "knowledge_base_ids": ["kb_node_002"],
                    "knowledge_base_items": [{"description": "Hoan Kiem Lake in the heart of Hanoi."}]
                }],
                "segment_captions": []
            }]
        },
    ]
}

MINI_MAP = {
    "Pen Tower": {"kb_id": "pen_tower", "in_kb": True},
    "Banh mi":   {"kb_id": None,        "in_kb": False},
    "Hoan Kiem Lake": {"kb_id": "hoan_kiem_lake", "in_kb": True},
}


def _write_files(tmp_path, dataset=MINI_DATASET, lmap=MINI_MAP):
    ds_path = tmp_path / "dataset.json"
    map_path = tmp_path / "landmark_map.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")
    map_path.write_text(json.dumps(lmap), encoding="utf-8")
    return ds_path, map_path


def test_in_kb_segments_extracted(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=10, seed=42)
    assert len(result["in_kb"]) == 2
    kb_ids = {v["gt_segments"][0]["kb_id"] for v in result["in_kb"]}
    assert kb_ids == {"pen_tower", "hoan_kiem_lake"}


def test_out_of_kb_segments_extracted(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=10, seed=42)
    assert len(result["out_of_kb"]) == 1
    assert result["out_of_kb"][0]["video_id"] == "vid2"


def test_reference_caption_populated(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=10, seed=42)
    seg = result["in_kb"][0]["gt_segments"][0]
    assert seg["reference_caption"]
    assert seg["kb_description"]


def test_sample_respects_total(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=1, seed=42)
    assert len(result["in_kb"]) <= 1


def test_segments_without_combined_caption_excluded(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    dataset = {
        "videos": [{
            "id": "vid_empty", "filename": "D.MOV", "duration": 10.0, "fps": 30, "height": 1080, "width": 1920,
            "segments": [{
                "id": "sx", "name": "Pen Tower", "start_time": 0.0, "end_time": 10.0, "duration": 10.0,
                "regions": [],  # no regions, no caption
                "segment_captions": []
            }]
        }]
    }
    ds, lmap = _write_files(tmp_path, dataset=dataset)
    result = build_test_set(ds, lmap, total_videos=10, seed=42)
    assert len(result["in_kb"]) == 0
