"""Two-stage KB matching: cosine retrieval + VLM re-rank."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
from PIL import Image

from hanoi_caption.image_describer import registry as _img_registry  # noqa: F401  ensures VLM is registered
from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
from hanoi_caption.kb_indexer import KBIndex, embed_text
from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import KBNode, MatchCandidate, MatchResult

DEFAULT_THRESHOLD = 0.45
TOPK = 3


def _default_embed(text: str) -> np.ndarray:
    return embed_text([text])[0]


def _vlm_rerank(
    image: Image.Image,
    candidates: list[MatchCandidate],
    kb_nodes: dict[str, KBNode],
) -> tuple[str | None, float]:
    """Ask the VLM to choose among the top-k or say 'none'.

    Returns (node_id_or_None, confidence_in_[0,1]).
    """
    import torch

    bundle = registry.get(VLM_NAME)
    processor, model = bundle["processor"], bundle["model"]

    options_block = "\n".join(
        f"- id: {c.node_id} | name: {kb_nodes[c.node_id].name_en} | "
        f"visual cues: {kb_nodes[c.node_id].visual_cues_en[:300]}"
        for c in candidates
    )
    prompt = (
        "You are a Hanoi tour expert. Choose which landmark this image shows. "
        "Reply with strict JSON: {\"node_id\": <id-or-null>, \"confidence\": <0-1>}.\n\n"
        f"Options:\n{options_block}\n\n"
        "If none of the options match the image, return node_id=null."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    raw = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        node_id = parsed.get("node_id")
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        return (None, 0.0)
    if node_id not in {c.node_id for c in candidates}:
        return (None, confidence)
    return (node_id, confidence)


def match_kb(
    image: Image.Image,
    holistic_desc: str,
    kb_index: KBIndex,
    kb_nodes: dict[str, KBNode],
    threshold: float = DEFAULT_THRESHOLD,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    rerank_fn: Callable[
        [Image.Image, list[MatchCandidate], dict[str, KBNode]],
        tuple[str | None, float],
    ]
    | None = None,
) -> MatchResult:
    embed_fn = embed_fn or _default_embed
    rerank_fn = rerank_fn or _vlm_rerank

    q = embed_fn(holistic_desc)
    candidates = kb_index.topk(q, k=TOPK)

    if not candidates or candidates[0].score < threshold:
        return MatchResult(node_id=None, confidence=candidates[0].score if candidates else 0.0, top_k=candidates)

    # The injected fake_rerank in tests has signature (image, candidate_ids, kb).
    # Real _vlm_rerank takes the same arity but with full candidates; pass full list.
    chosen, conf = rerank_fn(image, candidates, kb_nodes)
    return MatchResult(node_id=chosen, confidence=conf, top_k=candidates)
