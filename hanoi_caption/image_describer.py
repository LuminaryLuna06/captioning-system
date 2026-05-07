"""Whole-image describer: produce a holistic visual description without naming landmarks."""
from __future__ import annotations

from PIL import Image

from hanoi_caption.model_registry import registry

MODEL_NAME = "qwen25_vl_7b"
HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "Describe what is visually present in this image. "
    "Mention architecture, materials, layout, surroundings, and people. "
    "Use 3 to 5 sentences. "
    "Do NOT name specific landmarks, places, cities, or countries."
)


def _load():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(HF_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        load_in_4bit=True,
    )
    model.eval()
    return {"processor": processor, "model": model}


registry.register(MODEL_NAME, _load)


def describe_image(image: Image.Image) -> str:
    import torch

    bundle = registry.get(MODEL_NAME)
    processor, model = bundle["processor"], bundle["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    return decoded.strip()
