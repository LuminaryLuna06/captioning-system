"""Compose the final 150-300 word tour-guide caption."""
from __future__ import annotations

from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import KBNode, RegionDescription

MODEL_NAME = "qwen25_7b_instruct"
HF_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a tour guide writing for travelers. Voice: warm, observant, concrete. "
    "Write ONE paragraph of 150 to 300 words in English. "
    "Weave together (a) the historical and cultural facts provided and "
    "(b) the specific visual details actually observed in this photo. "
    "Do not invent facts beyond what is provided. Do not list — write prose. "
    "Do not mention that you are using a knowledge base or AI."
)


def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        HF_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        quantization_config=quant,
    )
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


registry.register(MODEL_NAME, _load)


def build_user_prompt(
    kb_node: KBNode,
    region_descriptions: list[RegionDescription],
    holistic_desc: str,
) -> str:
    if region_descriptions:
        regions_block = "\n".join(
            f"- ({rd.query}) {rd.text}" for rd in region_descriptions
        )
    else:
        regions_block = "- (none — phase 1, no region grounding yet)"
    return (
        f"Landmark: {kb_node.name_en}\n\n"
        f"Background:\n{kb_node.description_en}\n\n"
        f"What is visible in this photo:\n{regions_block}\n\n"
        f"Holistic view: {holistic_desc}\n\n"
        "Write the paragraph now."
    )


def compose(
    kb_node: KBNode,
    region_descriptions: list[RegionDescription],
    holistic_desc: str,
) -> str:
    import torch

    bundle = registry.get(MODEL_NAME)
    tok, model = bundle["tokenizer"], bundle["model"]

    user_prompt = build_user_prompt(kb_node, region_descriptions, holistic_desc)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tok.decode(
        out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
