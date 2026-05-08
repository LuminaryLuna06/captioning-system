from hanoi_caption.composer import build_user_prompt
from hanoi_caption.schemas import KBNode, RegionDescription


def _node():
    return KBNode(
        id="x", name_en="X Temple", name_vi="X", type="object",
        parent_id=None, description_en="X is old.", description_vi="",
        visual_cues_en="stone gate", visual_cues_vi="", tags=[],
    )


def test_prompt_includes_landmark_and_holistic():
    p = build_user_prompt(_node(), [], "A stone building with a courtyard.")
    assert "X Temple" in p
    assert "X is old." in p
    assert "A stone building" in p


def test_prompt_handles_empty_regions():
    p = build_user_prompt(_node(), [], "holistic")
    assert "phase 1" in p


def test_prompt_lists_regions():
    rds = [RegionDescription(query="gate", text="A red gate.")]
    p = build_user_prompt(_node(), rds, "holistic")
    assert "(gate) A red gate." in p
