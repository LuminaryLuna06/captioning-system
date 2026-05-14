from hanoi_caption.schemas import KBNode
from hanoi_caption.video_pipeline import dam_caption_segment


class _MockModel:
    def __init__(self):
        self.last_call = None

    def get_description(self, image_pil, mask_pil, query, **kwargs):
        # DAM's signature: lists for multi-image input
        self.last_call = {
            "image_pil": image_pil,
            "mask_pil": mask_pil,
            "query": query,
            "kwargs": kwargs,
        }
        return "  a caption  "


def _node():
    return KBNode(
        id="x", kb_id="temple_of_literature",
        name_en="Temple of Literature", name_vi="t",
        type="object", parent_id=None,
        description_en="A famous temple.", description_vi="",
        visual_cues_en="red roof, wooden columns", visual_cues_vi="",
        tags=[],
    )


def test_prompt_includes_one_image_token_per_frame():
    model = _MockModel()
    frames = ["frame1", "frame2", "frame3"]  # opaque to the function

    dam_caption_segment(
        model=model,
        frames=frames,
        node=_node(),
        full_image_mask_fn=lambda f: f"mask_of_{f}",
        image_token="<image>",
    )

    # 3 frames -> 3 <image> tokens in the assembled prompt
    assert model.last_call["query"].count("<image>") == 3


def test_prompt_includes_landmark_name_and_kb_facts():
    model = _MockModel()
    dam_caption_segment(
        model=model,
        frames=["f"],
        node=_node(),
        full_image_mask_fn=lambda f: "m",
        image_token="<image>",
    )
    q = model.last_call["query"]
    assert "Temple of Literature" in q
    assert "A famous temple." in q
    assert "red roof, wooden columns" in q


def test_passes_frames_and_masks_as_parallel_lists():
    model = _MockModel()
    frames = ["a", "b", "c"]
    dam_caption_segment(
        model=model, frames=frames, node=_node(),
        full_image_mask_fn=lambda f: f"M:{f}",
        image_token="<image>",
    )
    assert model.last_call["image_pil"] == ["a", "b", "c"]
    assert model.last_call["mask_pil"] == ["M:a", "M:b", "M:c"]


def test_returns_stripped_caption():
    model = _MockModel()
    out = dam_caption_segment(
        model=model, frames=["f"], node=_node(),
        full_image_mask_fn=lambda f: "m",
        image_token="<image>",
    )
    assert out == "a caption"
