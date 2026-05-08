from PIL import Image

from hanoi_caption.pipeline import caption_phase1
from hanoi_caption.schemas import KBNode, MatchCandidate, MatchResult


def _img():
    return Image.new("RGB", (8, 8), color=(0, 0, 0))


def _kb():
    return {
        "a": KBNode(
            id="a", name_en="A", name_vi="A", type="object", parent_id=None,
            description_en="A description.", description_vi="",
            visual_cues_en="cue", visual_cues_vi="", tags=[],
        )
    }


def test_phase1_returns_caption_when_match_succeeds():
    res = caption_phase1(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,  # ignored when describe/match are mocked
        describe_fn=lambda im: "a black square",
        match_fn=lambda im, desc, idx, kb: MatchResult(
            node_id="a", confidence=0.9,
            top_k=[MatchCandidate(node_id="a", score=0.9)],
        ),
        compose_fn=lambda node, regions, desc: "A long paragraph about A.",
    )
    assert res.caption == "A long paragraph about A."
    assert res.refusal is None
    assert res.debug["match"]["node_id"] == "a"


def test_phase1_refuses_when_no_match():
    res = caption_phase1(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,
        describe_fn=lambda im: "?",
        match_fn=lambda im, desc, idx, kb: MatchResult(
            node_id=None, confidence=0.1, top_k=[],
        ),
        compose_fn=lambda *a, **k: "should not be called",
    )
    assert res.caption is None
    assert "Not a recognized" in res.refusal
