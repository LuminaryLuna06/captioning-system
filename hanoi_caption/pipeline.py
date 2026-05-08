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

    node = kb_nodes[match.node_id]
    caption = compose_fn(node, [], holistic)
    debug["caption_chars"] = len(caption)
    return CaptionResult(caption=caption, refusal=None, debug=debug)
