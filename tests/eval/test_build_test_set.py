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

MINI_KB = [
    {
        "id": "kb_node_001", "kb_id": "pen_tower", "type": "object",
        "name": "Pen Tower", "name_vi": "Thap But",
        "description": "Five-story stone tower built in 1865.", "description_vi": "",
        "visual_cues": "A tall stone tower.", "visual_cues_vi": "",
    },
    {
        "id": "kb_node_002", "kb_id": "hoan_kiem_lake", "type": "object",
        "name": "Hoan Kiem Lake", "name_vi": "Ho Hoan Kiem",
        "description": "Hoan Kiem Lake in the heart of Hanoi.", "description_vi": "",
        "visual_cues": "A body of water.", "visual_cues_vi": "",
    },
]


def _write_files(tmp_path, dataset=MINI_DATASET, lmap=MINI_MAP, kb=MINI_KB):
    ds_path = tmp_path / "dataset.json"
    map_path = tmp_path / "landmark_map.json"
    kb_path = tmp_path / "kb.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")
    map_path.write_text(json.dumps(lmap), encoding="utf-8")
    kb_path.write_text(json.dumps(kb), encoding="utf-8")
    return ds_path, map_path, kb_path


def test_in_kb_segments_extracted(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap, kb = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=10, seed=42, kb_path=kb)
    assert len(result["in_kb"]) == 2
    kb_ids = {v["gt_segments"][0]["kb_id"] for v in result["in_kb"]}
    assert kb_ids == {"pen_tower", "hoan_kiem_lake"}


def test_out_of_kb_segments_extracted(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap, kb = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=10, seed=42, kb_path=kb)
    assert len(result["out_of_kb"]) == 1
    assert result["out_of_kb"][0]["video_id"] == "vid2"


def test_reference_caption_populated(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap, kb = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=10, seed=42, kb_path=kb)
    seg = result["in_kb"][0]["gt_segments"][0]
    assert seg["reference_caption"]
    assert seg["kb_description"]


def test_sample_respects_total(tmp_path):
    from scripts.eval.build_test_set import build_test_set
    ds, lmap, kb = _write_files(tmp_path)
    result = build_test_set(ds, lmap, total_videos=1, seed=42, kb_path=kb)
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
    ds, lmap, kb = _write_files(tmp_path, dataset=dataset)
    result = build_test_set(ds, lmap, total_videos=10, seed=42, kb_path=kb)
    assert len(result["in_kb"]) == 0


def test_video_with_multiple_landmarks_appears_per_landmark(tmp_path):
    """A video with 2 in-KB landmark segments should contribute one entry per segment."""
    from scripts.eval.build_test_set import build_test_set
    dataset = {
        "videos": [{
            "id": "multi_vid", "filename": "M.MOV", "duration": 60.0,
            "fps": 30, "height": 1080, "width": 1920,
            "segments": [
                {
                    "id": "seg_a", "name": "Pen Tower",
                    "start_time": 0.0, "end_time": 30.0, "duration": 30.0,
                    "regions": [{"captions": {"en": {"combined": "Caption A."}},
                                 "knowledge_base_ids": ["id1"],
                                 "knowledge_base_items": [{"description": "KB A"}]}],
                    "segment_captions": []
                },
                {
                    "id": "seg_b", "name": "Hoan Kiem Lake",
                    "start_time": 30.0, "end_time": 60.0, "duration": 30.0,
                    "regions": [{"captions": {"en": {"combined": "Caption B."}},
                                 "knowledge_base_ids": ["id2"],
                                 "knowledge_base_items": [{"description": "KB B"}]}],
                    "segment_captions": []
                },
            ]
        }]
    }
    lmap = {
        "Pen Tower": {"kb_id": "pen_tower", "in_kb": True},
        "Hoan Kiem Lake": {"kb_id": "hoan_kiem_lake", "in_kb": True},
    }
    ds_path = tmp_path / "ds.json"
    map_path = tmp_path / "map.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")
    map_path.write_text(json.dumps(lmap), encoding="utf-8")
    result = build_test_set(ds_path, map_path, total_videos=10, seed=42)
    assert len(result["in_kb"]) == 2
    kb_ids = {e["gt_segments"][0]["kb_id"] for e in result["in_kb"]}
    assert kb_ids == {"pen_tower", "hoan_kiem_lake"}
