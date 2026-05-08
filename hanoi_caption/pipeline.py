"""End-to-end orchestration."""
from __future__ import annotations

from typing import Callable

from PIL import Image

from hanoi_caption.kb_indexer import KBIndex
from hanoi_caption.schemas import (
    CaptionResult,
    KBNode,
    MatchResult,
    RegionDescription,
)

REFUSAL_TEXT = "Not a recognized Hanoi landmark."


def caption_phase1(
    image: Image.Image,
    kb_nodes: dict[str, KBNode],
    kb_index: KBIndex | None,
    describe_fn: Callable[[Image.Image], str] | None = None,
    match_fn: Callable[
        [Image.Image, str, KBIndex | None, dict[str, KBNode]], MatchResult
    ]
    | None = None,
    compose_fn: Callable[
        [KBNode, list[RegionDescription], str], str
    ]
    | None = None,
) -> CaptionResult:
    """KB-only path: image → describe → match → compose. No DAM/SAM/GroundingDINO."""
    if describe_fn is None:
        from hanoi_caption.image_describer import describe_image as describe_fn  # noqa
    if match_fn is None:
        from hanoi_caption.kb_matcher import match_kb as _match
        match_fn = lambda im, desc, idx, kb: _match(im, desc, idx, kb)
    if compose_fn is None:
        from hanoi_caption.composer import compose as compose_fn  # noqa

    debug: dict = {}

    holistic = describe_fn(image)
    debug["holistic_desc"] = holistic

    match = match_fn(image, holistic, kb_index, kb_nodes)
    debug["match"] = match.model_dump()

    if match.node_id is None:
        return CaptionResult(caption=None, refusal=REFUSAL_TEXT, debug=debug)

    # The VLM finished its job after match; evict it so the composer's 4-bit
    # load doesn't double-peak VRAM on a 16 GB GPU. Mirrors caption_phase2.
    try:
        from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
        from hanoi_caption.model_registry import registry
        registry.evict(VLM_NAME)
    except Exception:
        pass

    node = kb_nodes[match.node_id]
    caption = compose_fn(node, [], holistic)
    debug["caption_chars"] = len(caption)
    return CaptionResult(caption=caption, refusal=None, debug=debug)


from hanoi_caption.schemas import Region


def caption_phase2(
    image: Image.Image,
    kb_nodes: dict[str, KBNode],
    kb_index: KBIndex | None,
    describe_fn: Callable[[Image.Image], str] | None = None,
    match_fn: Callable[
        [Image.Image, str, KBIndex | None, dict[str, KBNode]], MatchResult
    ]
    | None = None,
    extract_queries_fn: Callable[[str], list[str]] | None = None,
    propose_regions_fn: Callable[[Image.Image, list[str]], list[Region]] | None = None,
    describe_regions_fn: Callable[
        [Image.Image, list[Region]], list[RegionDescription]
    ]
    | None = None,
    compose_fn: Callable[
        [KBNode, list[RegionDescription], str], str
    ]
    | None = None,
) -> CaptionResult:
    """Full pipeline: identify -> queries -> detect/segment/describe -> compose.

    Manages model eviction between stages to stay under the 14 GB working-set budget.
    """
    if describe_fn is None:
        from hanoi_caption.image_describer import describe_image as describe_fn  # noqa
    if match_fn is None:
        from hanoi_caption.kb_matcher import match_kb as _match
        match_fn = lambda im, desc, idx, kb: _match(im, desc, idx, kb)
    if extract_queries_fn is None:
        from hanoi_caption.query_extractor import extract_queries as extract_queries_fn  # noqa
    if propose_regions_fn is None:
        from hanoi_caption.region_proposer import propose_regions as propose_regions_fn  # noqa
    if describe_regions_fn is None:
        from hanoi_caption.region_describer import describe_regions as describe_regions_fn  # noqa
    if compose_fn is None:
        from hanoi_caption.composer import compose as compose_fn  # noqa

    from hanoi_caption.model_registry import registry

    debug: dict = {}

    # Stage 3 — describe
    holistic = describe_fn(image)
    debug["holistic_desc"] = holistic

    # Stage 4 — match (uses VLM re-rank, then we are done with the VLM)
    match = match_fn(image, holistic, kb_index, kb_nodes)
    debug["match"] = match.model_dump()
    if match.node_id is None:
        return CaptionResult(caption=None, refusal=REFUSAL_TEXT, debug=debug)

    node = kb_nodes[match.node_id]

    # Stage 5 — extract queries (loads Qwen2.5-7B; keep it for stage 8)
    queries = extract_queries_fn(node.visual_cues_en)
    debug["queries"] = queries

    # Free the VLM before loading detection stack
    try:
        from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
        registry.evict(VLM_NAME)
    except Exception:
        pass

    # Stage 6 — propose regions
    regions = propose_regions_fn(image, queries)
    debug["n_regions"] = len(regions)
    debug["regions"] = [r.model_dump(exclude={"mask_png_b64"}) for r in regions]

    # Stage 7 — describe regions
    region_descs = describe_regions_fn(image, regions)
    debug["region_descriptions"] = [rd.model_dump() for rd in region_descs]

    # Free detection stack before composing
    for n in ("grounding_dino", "sam2", "dam_3b", "bge_m3"):
        try:
            registry.evict(n)
        except Exception:
            pass

    # Stage 8 — compose
    caption = compose_fn(node, region_descs, holistic)
    debug["caption_chars"] = len(caption)
    return CaptionResult(caption=caption, refusal=None, debug=debug)
