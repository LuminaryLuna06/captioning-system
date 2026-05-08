"""Convert KB Visual Cues into short detector-friendly noun phrases."""
from __future__ import annotations

import json
import re

from hanoi_caption.composer import MODEL_NAME as LLM_NAME
from hanoi_caption.composer import _load as _llm_load  # noqa: F401  ensures registration
from hanoi_caption.model_registry import registry

EXTRACT_PROMPT = (
    "Extract 4 to 8 short noun phrases (1 to 4 words each) from the description below. "
    "Each phrase MUST name a physically detectable object that an open-vocabulary "
    "object detector can find in a photo (e.g., 'red gate', 'stone stele', 'tiered roof'). "
    "No verbs. No abstractions. No place names.\n\n"
    "Reply with ONLY a JSON array of strings. No prose, no markdown.\n\n"
    "Description:\n{desc}"
)


def parse_queries(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\[.*?\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    seen = set()
    out: list[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        norm = it.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def extract_queries(visual_cues_text: str) -> list[str]:
    import torch

    bundle = registry.get(LLM_NAME)
    tok, model = bundle["tokenizer"], bundle["model"]
    messages = [
        {"role": "user", "content": EXTRACT_PROMPT.format(desc=visual_cues_text)}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    raw = tok.decode(
        out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return parse_queries(raw)
