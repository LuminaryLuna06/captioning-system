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
