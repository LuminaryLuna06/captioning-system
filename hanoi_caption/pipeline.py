"""End-to-end orchestration."""
from __future__ import annotations

import time
from contextlib import contextmanager
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


@contextmanager
def _stage_timer(debug: dict, name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        debug.setdefault("timings", {})[name] = time.perf_counter() - t0


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

    # 3B VLM (~2 GB) coexists comfortably with the 4-bit composer LLM (~5 GB),
    # so no eviction needed — VLM stays resident for the next call.

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

    # Resize before any stage runs: high-res photos blow up SAM2 AMG latency.
    image = image.copy()
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    debug: dict = {"timings": {}}

    # Stage 3 — describe
    with _stage_timer(debug, "describe"):
        holistic = describe_fn(image)
    debug["holistic_desc"] = holistic

    # Stage 4 — match (uses VLM re-rank, then we are done with the VLM)
    with _stage_timer(debug, "match"):
        match = match_fn(image, holistic, kb_index, kb_nodes)
    debug["match"] = match.model_dump()
    if match.node_id is None:
        return CaptionResult(caption=None, refusal=REFUSAL_TEXT, debug=debug)

    node = kb_nodes[match.node_id]

    # Stage 5 — extract queries (loads Qwen2.5-7B; keep it for stage 8)
    with _stage_timer(debug, "extract_queries"):
        queries = extract_queries_fn(node.visual_cues_en)
    debug["queries"] = queries

    # Stage 6 — propose regions
    with _stage_timer(debug, "propose_regions"):
        regions = propose_regions_fn(image, queries)
    debug["n_regions"] = len(regions)
    debug["regions"] = [r.model_dump(exclude={"mask_png_b64"}) for r in regions]

    # Free everything except the composer LLM before DAM (~6 GB) loads.
    # Empirically, keeping VLM/GDINO/SAM2/BGE resident on a 16 GB GPU pushes
    # peak VRAM past the Windows TDR limit and crashes CUDA (segfault).
    # 3B VLM still wins us speed: it re-loads in ~10 s vs the 7B's ~30 s.
    from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
    for n in (VLM_NAME, "grounding_dino", "sam2", "bge_m3"):
        try:
            registry.evict(n)
        except Exception:
            pass

    # Stage 7 — describe regions
    with _stage_timer(debug, "describe_regions"):
        region_descs = describe_regions_fn(image, regions)
    debug["region_descriptions"] = [rd.model_dump() for rd in region_descs]

    # DAM (~6 GB) freed for the composer's activation headroom.
    try:
        registry.evict("dam_3b")
    except Exception:
        pass

    # Stage 8 — compose
    with _stage_timer(debug, "compose"):
        caption = compose_fn(node, region_descs, holistic)
    debug["caption_chars"] = len(caption)
    return CaptionResult(caption=caption, refusal=None, debug=debug)
