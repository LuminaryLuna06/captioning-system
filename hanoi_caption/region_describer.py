"""DAM-3B focal description of masked regions.

DAM is installed from NVIDIA's repo:
    pip install git+https://github.com/NVlabs/describe-anything.git

The exact import path / API may differ between releases. The reference
quickstart at the time of writing exposes a high-level `DAMModel.describe`
that takes (PIL.Image, mask: np.ndarray, prompt: str | None) and returns
a string. If the API has changed by your install date, adapt _load and
describe_region to the current API. Treat that adaptation as part of
this task — do not invent helpers.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import Region, RegionDescription

MODEL_NAME = "dam_3b"
DAM_PROMPT = (
    "Describe the highlighted region in 1 to 2 short sentences. "
    "Focus on visible attributes (material, color, condition, posture, action). "
    "Do not name the location or invent context outside the region."
)


def _load():
    # Adapt this import to the installed DAM release.
    from dam import DAMModel  # type: ignore

    model = DAMModel.from_pretrained("nvidia/DAM-3B")
    model.eval()
    model.to("cuda")
    return model


registry.register(MODEL_NAME, _load)


def _b64_png_to_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return (np.array(img) > 127).astype(np.uint8)


def describe_regions(image: Image.Image, regions: list[Region]) -> list[RegionDescription]:
    if not regions:
        return []
    model = registry.get(MODEL_NAME)
    out: list[RegionDescription] = []
    for r in regions:
        mask = _b64_png_to_mask(r.mask_png_b64)
        text = model.describe(image=image, mask=mask, prompt=DAM_PROMPT)
        out.append(RegionDescription(query=r.query, text=text.strip()))
    return out
