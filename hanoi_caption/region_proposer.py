"""Detect KB-driven regions and segment them.

Grounding DINO (HuggingFace transformers integration) provides text-prompted
detection. SAM 2 converts boxes to binary masks. We filter masks for size
and IoU before passing to DAM.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from hanoi_caption.model_registry import registry

GDINO_NAME = "grounding_dino"
GDINO_HF = "IDEA-Research/grounding-dino-base"
SAM2_NAME = "sam2"
SAM2_HF = "facebook/sam2-hiera-base-plus"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
MIN_AREA_FRAC = 0.01
IOU_THRESHOLD = 0.7
MAX_KEEP = 2


def _load_gdino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(GDINO_HF)
    # GDINO is small (~340M params); fp16 triggers vision/text dtype mismatches
    # in transformers ≥4.46. Load fp32 — adds ~700MB VRAM, still well inside
    # the spec §10.1 stage-6 budget (~10 GB peak).
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_HF).to("cuda")
    model.eval()
    return {"processor": processor, "model": model}


def _load_sam2():
    # Uses Meta's sam2 package; install from PyPI as `sam2`.
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(SAM2_HF)
    return predictor


registry.register(GDINO_NAME, _load_gdino)
registry.register(SAM2_NAME, _load_sam2)


@dataclass
class _Detection:
    box: tuple[float, float, float, float]
    score: float
    query: str


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def filter_regions(
    detections: list,
    image_area: float,
    min_area_frac: float = MIN_AREA_FRAC,
    iou_threshold: float = IOU_THRESHOLD,
    max_keep: int = MAX_KEEP,
) -> list:
    survivors = [d for d in detections if _box_area(d.box) / image_area >= min_area_frac]
    survivors.sort(key=lambda d: d.score, reverse=True)
    kept: list = []
    for d in survivors:
        if any(_iou(d.box, k.box) > iou_threshold for k in kept):
            continue
        kept.append(d)
        if len(kept) >= max_keep:
            break
    return kept


def _detect(image: Image.Image, queries: list[str]) -> list[_Detection]:
    if not queries:
        return []
    bundle = registry.get(GDINO_NAME)
    processor, model = bundle["processor"], bundle["model"]

    text_prompt = ". ".join(q.lower().strip() for q in queries) + "."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    # transformers ≥4.46 renamed `box_threshold` to `threshold` and now returns
    # the detector's text labels under `text_labels` (`labels` becomes a tensor
    # of class indices, not strings).
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    out: list[_Detection] = []
    text_labels = results.get("text_labels", results.get("labels"))
    for box, score, label in zip(results["boxes"], results["scores"], text_labels):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        out.append(_Detection(box=(x1, y1, x2, y2), score=float(score), query=str(label)))
    return out


def _segment(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
    if not boxes:
        return []
    predictor = registry.get(SAM2_NAME)
    image_np = np.array(image)
    predictor.set_image(image_np)
    masks: list[np.ndarray] = []
    for box in boxes:
        m, _, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
        masks.append(m[0].astype(np.uint8))
    return masks


def _sam_automask_topk(image: Image.Image, k: int = 4) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    predictor = registry.get(SAM2_NAME)
    # points_per_side=16 (default 32) cuts AMG compute ~4x for the fallback path.
    gen = SAM2AutomaticMaskGenerator(predictor.model, points_per_side=16)
    masks = gen.generate(np.array(image))
    masks.sort(key=lambda m: m["area"], reverse=True)
    out = []
    for m in masks[:k]:
        x, y, w, h = m["bbox"]
        out.append((m["segmentation"].astype(np.uint8), (x, y, x + w, y + h)))
    return out


def _mask_to_b64_png(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def propose_regions(image: Image.Image, queries: list[str]):
    """Returns list[Region] (Pydantic) — see hanoi_caption.schemas."""
    from hanoi_caption.schemas import Region

    image_area = image.size[0] * image.size[1]

    dets = _detect(image, queries)
    kept = filter_regions(dets, image_area=image_area)

    if kept:
        masks = _segment(image, [d.box for d in kept])
        out: list[Region] = []
        for d, m in zip(kept, masks):
            out.append(Region(
                box=d.box, mask_png_b64=_mask_to_b64_png(m),
                query=d.query, score=d.score,
            ))
        return out

    # Fallback: SAM auto-mask top-K
    masks_with_boxes = _sam_automask_topk(image, k=4)
    return [
        Region(
            box=box, mask_png_b64=_mask_to_b64_png(m),
            query="(automask)", score=0.0,
        )
        for m, box in masks_with_boxes
    ]
