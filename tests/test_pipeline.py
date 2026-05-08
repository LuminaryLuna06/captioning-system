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


from hanoi_caption.pipeline import caption_phase2
from hanoi_caption.schemas import Region, RegionDescription


def test_phase2_evicts_models_in_order_and_returns_caption():
    calls: list[str] = []

    def describe_fn(im):
        calls.append("describe"); return "holistic"

    def match_fn(im, desc, idx, kb):
        calls.append("match"); return MatchResult(
            node_id="a", confidence=0.9,
            top_k=[MatchCandidate(node_id="a", score=0.9)],
        )

    def extract_fn(text):
        calls.append("extract"); return ["q1", "q2"]

    def propose_fn(im, queries):
        calls.append("propose")
        return [Region(box=(0,0,10,10), mask_png_b64="", query="q1", score=0.9)]

    def regions_describe_fn(im, regions):
        calls.append("regdesc")
        return [RegionDescription(query="q1", text="a thing")]

    def compose_fn(node, regions, desc):
        calls.append("compose")
        return "A " + " ".join(r.text for r in regions) + " paragraph."

    res = caption_phase2(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,
        describe_fn=describe_fn,
        match_fn=match_fn,
        extract_queries_fn=extract_fn,
        propose_regions_fn=propose_fn,
        describe_regions_fn=regions_describe_fn,
        compose_fn=compose_fn,
    )
    assert res.caption is not None and "a thing" in res.caption
    assert calls == ["describe", "match", "extract", "propose", "regdesc", "compose"]


def test_phase2_refusal_path_skips_dam():
    calls: list[str] = []

    def fail(*a, **k):
        calls.append("should_not_be_called"); raise AssertionError

    res = caption_phase2(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,
        describe_fn=lambda im: "?",
        match_fn=lambda im, d, idx, kb: MatchResult(node_id=None, confidence=0.0, top_k=[]),
        extract_queries_fn=fail,
        propose_regions_fn=fail,
        describe_regions_fn=fail,
        compose_fn=fail,
    )
    assert res.caption is None and "Not a recognized" in res.refusal
    assert calls == []
