"""DAM-3B focal description of masked regions.

DAM is installed from NVIDIA's repo:
    pip install git+https://github.com/NVlabs/describe-anything.git

The plan originally targeted a hypothetical `dam.DAMModel.from_pretrained` /
`.describe` API. The shipped `dam` package (current at the time of this run)
exposes `DescribeAnythingModel` instead, takes a *local checkpoint dir* (not a
HF model id) for `model_path`, requires explicit `conv_mode` + `prompt_mode`
constructor args, and offers `get_description(image_pil, mask_pil, query=...)`
where `mask_pil` is a PIL Image. This module wraps that real API.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import Region, RegionDescription

MODEL_NAME = "dam_3b"
DAM_REPO = "nvidia/DAM-3B"
DAM_CONV_MODE = "v1"
DAM_PROMPT_MODE = "full+focal_crop"
DAM_PROMPT = (
    "Describe the highlighted region in 1 to 2 short sentences. "
    "Focus on visible attributes (material, color, condition, posture, action). "
    "Do not name the location or invent context outside the region."
)


def _load():
    # --- compatibility shims for transformers ≥5.0 ↔ dam ≤current ---
    # 1. transformers 5.x removed `no_init_weights` from `modeling_utils`;
    #    the installed `dam` package still imports it from the old path.
    # 2. transformers 5.x added `mark_tied_weights_as_initialized` that
    #    iterates `self.all_tied_weights_keys.keys()`, but DAM-bundled
    #    sub-models (e.g. MultimodalProjector) only define the older
    #    `_tied_weights_keys` list. Default the new attribute to {} so the
    #    iteration is a no-op — tied-weights bookkeeping is diagnostic only
    #    and does not affect inference correctness.
    import transformers.modeling_utils as _mu
    if not hasattr(_mu, "no_init_weights"):
        from transformers.initialization import no_init_weights as _niw
        _mu.no_init_weights = _niw
    if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
        _mu.PreTrainedModel.all_tied_weights_keys = {}

    from huggingface_hub import snapshot_download
    from dam import DescribeAnythingModel  # type: ignore

    # DAM expects a *local* checkpoint dir. snapshot_download (no local_dir)
    # places the snapshot in the standard HF cache and returns its path —
    # no extra disk copy.
    model_path = snapshot_download(DAM_REPO)

    model = DescribeAnythingModel(
        model_path=model_path,
        conv_mode=DAM_CONV_MODE,
        prompt_mode=DAM_PROMPT_MODE,
    )
    model.eval()
    model.to("cuda")
    return model


registry.register(MODEL_NAME, _load)


def _b64_png_to_mask_pil(b64: str) -> Image.Image:
    """Decode the base64 PNG mask back to a binary L-mode PIL Image (0/255)."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    arr = (np.array(img) > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def describe_regions(image: Image.Image, regions: list[Region]) -> list[RegionDescription]:
    if not regions:
        return []
    model = registry.get(MODEL_NAME)
    # DAM's conv template requires the literal <image> token in the query so
    # the prompt builder knows where the visual content goes.
    from dam import DEFAULT_IMAGE_TOKEN  # type: ignore
    query = f"{DEFAULT_IMAGE_TOKEN}\n{DAM_PROMPT}"
    out: list[RegionDescription] = []
    for r in regions:
        mask_pil = _b64_png_to_mask_pil(r.mask_png_b64)
        text = model.get_description(image, mask_pil, query=query)
        out.append(RegionDescription(query=r.query, text=text.strip()))
    return out
