"""Retriever pipeline: VLM describe -> cosine + VLM rerank -> DAM caption.

Flow:
    Stage A — VLM produces a holistic description (the cosine retrieval query).
    Stage B — match: BGE-M3 cosine top-3 + VLM rerank picks the landmark.
              The rerank guards against the failure mode where a confident
              cosine score lands on the wrong landmark (e.g. Opera House on
              a Temple photo at 0.67 cosine).
    Stage C — DAM-3B generates the final caption with the matched landmark's
              KB facts embedded directly in its prompt.

Drops query extraction, region proposal, per-region DAM, and the composer
LLM in favor of a single DAM call. Trade-off: faster, but the caption's
prose depends on DAM's underlying LLM rather than the 7B composer.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

import numpy as np
from PIL import Image

from hanoi_caption.kb_indexer import KBIndex
from hanoi_caption.kb_matcher import DEFAULT_THRESHOLD, match_kb as _match_kb
from hanoi_caption.model_registry import registry
from hanoi_caption.region_describer import MODEL_NAME as DAM_NAME
from hanoi_caption.schemas import CaptionResult, KBNode, MatchResult

REFUSAL_TEXT = "Not a recognized Hanoi landmark."

DAM_CAPTION_PROMPT = (
    "{image_token}\n"
    "This photograph shows {name}.\n\n"
    "Historical and cultural context:\n{description}\n\n"
    "Notable visual features that may be present: {visual_cues}\n\n"
    "Write ONE warm, observant tour-guide paragraph (150 to 300 words, English) "
    "describing what you observe in the photo and weaving in the historical "
    "context above. Do not invent facts beyond what is provided. Write prose, "
    "not a list. Do not mention you are using a knowledge base or AI."
)


@contextmanager
def _stage_timer(debug: dict, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        debug.setdefault("timings", {})[name] = time.perf_counter() - t0


def _full_image_mask(image: Image.Image) -> Image.Image:
    arr = np.full((image.size[1], image.size[0]), 255, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _dam_caption_with_landmark(image: Image.Image, node: KBNode) -> str:
    from dam import DEFAULT_IMAGE_TOKEN  # type: ignore

    model = registry.get(DAM_NAME)
    mask = _full_image_mask(image)
    query = DAM_CAPTION_PROMPT.format(
        image_token=DEFAULT_IMAGE_TOKEN,
        name=node.name_en,
        description=node.description_en,
        visual_cues=node.visual_cues_en,
    )
    text = model.get_description(image, mask, query=query)
    return text.strip()


def _dam_describe_image(image: Image.Image) -> str:
    """Use DAM-3B for the initial holistic description (Stage A)."""
    from dam import DEFAULT_IMAGE_TOKEN  # type: ignore
    from hanoi_caption.image_describer import PROMPT

    model = registry.get(DAM_NAME)
    mask = _full_image_mask(image)
    query = f"{DEFAULT_IMAGE_TOKEN}\n{PROMPT}"
    text = model.get_description(image, mask, query=query)
    return text.strip()


def caption_retriever(
    image: Image.Image,
    kb_nodes: dict[str, KBNode],
    kb_index: KBIndex,
    threshold: float = DEFAULT_THRESHOLD,
    describe_fn: Callable[[Image.Image], str] | None = None,
    match_fn: Callable[
        [Image.Image, str, KBIndex | None, dict[str, KBNode]], MatchResult
    ] | None = None,
    dam_caption_fn: Callable[[Image.Image, KBNode], str] | None = None,
) -> CaptionResult:
    """Retriever pipeline: DAM describe -> match (cosine + rerank) -> DAM caption.

    Skips query extraction, region proposal, per-region DAM, and the composer
    LLM. Returns refusal when match_fn cannot identify a landmark.
    """
    if describe_fn is None:
        describe_fn = _dam_describe_image
    if match_fn is None:
        match_fn = lambda im, desc, idx, kb: _match_kb(im, desc, idx, kb, threshold=threshold)
    if dam_caption_fn is None:
        dam_caption_fn = _dam_caption_with_landmark

    image = image.copy()
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    debug: dict = {"timings": {}}

    # Stage A — DAM holistic description (consumed by match; not user-facing)
    with _stage_timer(debug, "describe"):
        holistic = describe_fn(image)
    debug["holistic_desc"] = holistic

    # Stage B — match (BGE cosine top-3 + VLM rerank picks final landmark)
    with _stage_timer(debug, "match"):
        match = match_fn(image, holistic, kb_index, kb_nodes)
    debug["match"] = match.model_dump()
    if match.node_id is None:
        return CaptionResult(caption=None, refusal=REFUSAL_TEXT, debug=debug)

    node = kb_nodes[match.node_id]

    # Free VLM (Qwen) + BGE-M3 before DAM Stage C — keeps peak under 16 GB.
    # DAM is already loaded from Stage A, so Stage C will be fast.
    # from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
    # for n in (VLM_NAME, "bge_m3"):
    #     try:
    #         registry.evict(n)
    #     except Exception:
    #         pass

    # Stage C — DAM produces the final caption with KB facts in its prompt
    with _stage_timer(debug, "dam_caption"):
        caption = dam_caption_fn(image, node)
    debug["caption_chars"] = len(caption)

    return CaptionResult(caption=caption, refusal=None, debug=debug)
