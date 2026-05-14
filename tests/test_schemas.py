import pytest
from pydantic import ValidationError

from hanoi_caption.schemas import (
    KBNode,
    MatchResult,
    Region,
    RegionDescription,
    CaptionResult,
)


def test_kbnode_minimal_object():
    node = KBNode(
        id="temple_of_literature",
        name_en="Temple of Literature",
        name_vi="Văn Miếu – Quốc Tử Giám",
        type="object",
        parent_id="categoryHaNoi",
        description_en="...",
        description_vi="...",
        visual_cues_en="stone steles, tiered roof",
        visual_cues_vi="...",
        tags=[],
    )
    assert node.id == "temple_of_literature"
    assert node.type == "object"


def test_kbnode_rejects_unknown_type():
    with pytest.raises(ValidationError):
        KBNode(
            id="x", name_en="x", name_vi="x", type="bogus",
            parent_id=None, description_en="x", description_vi="x",
            visual_cues_en="x", visual_cues_vi="x", tags=[],
        )


def test_match_result_none_path():
    r = MatchResult(node_id=None, confidence=0.0, top_k=[])
    assert r.node_id is None


def test_caption_result_either_caption_or_refusal():
    ok = CaptionResult(caption="A long paragraph...", refusal=None, debug={})
    assert ok.caption is not None
    refused = CaptionResult(caption=None, refusal="Not recognized.", debug={})
    assert refused.refusal is not None


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
