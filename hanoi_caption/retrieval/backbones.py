"""Pluggable image-embedding backbones for KB retrieval comparison.

Each concrete extractor implements the `BackboneExtractor` protocol:
- `name`: short id used as the cache subdirectory name
- `dim`: embedding dimension
- `extract(images)`: return float32 (N, dim) L2-normalized embeddings
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


@runtime_checkable
class BackboneExtractor(Protocol):
    name: str
    dim: int

    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return float32 array of shape (len(images), dim), L2-normalized along axis=1."""
        ...


class _HFExtractor:
    """Shared boilerplate: load HF processor + model on CUDA, run a dummy forward
    to detect `dim`, expose `extract(images)`.

    Subclasses set `name` + `_MODEL_ID` and override `_load_model()` (only when the
    default `AutoModel` isn't right) and `_embed(outputs) -> Tensor` to pick which
    output is the embedding."""

    name: str
    _MODEL_ID: str

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{type(self).__name__} requires CUDA (this project targets a single 16 GB RTX)"
            )
        self.device = torch.device("cuda")
        self.processor = AutoImageProcessor.from_pretrained(self._MODEL_ID)
        self.model = self._load_model().to(self.device).eval()
        self.dim = self._detect_dim()

    def _load_model(self):
        return AutoModel.from_pretrained(self._MODEL_ID)

    def _embed(self, outputs) -> torch.Tensor:
        raise NotImplementedError

    def _detect_dim(self) -> int:
        dummy = Image.new("RGB", (224, 224), color="gray")
        with torch.no_grad():
            inputs = self.processor(images=[dummy], return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            emb = self._embed(outputs)
        return int(emb.shape[-1])

    def extract(self, images):
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = self._embed(outputs)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)


class Dinov3Extractor(_HFExtractor):
    name = "dinov3_vits16"
    _MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

    def _embed(self, outputs):
        return outputs.last_hidden_state[:, 0]


class Dinov3LargeExtractor(_HFExtractor):
    name = "dinov3_vitl16"
    _MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

    def _embed(self, outputs):
        return outputs.last_hidden_state[:, 0]


class Resnet50Extractor(_HFExtractor):
    name = "resnet50"
    _MODEL_ID = "microsoft/resnet-50"

    def _embed(self, outputs):
        return outputs.pooler_output.flatten(1)


class Siglip2Extractor(_HFExtractor):
    name = "siglip2_base"
    _MODEL_ID = "google/siglip2-base-patch16-224"

    def _load_model(self):
        # google/siglip2-base-patch16-224 ships with a SiglipConfig (v1 architecture,
        # v2 weights/training), so Siglip2VisionModel.from_pretrained raises a shape
        # mismatch — fall back to SiglipVisionModel which is still vision-only.
        try:
            from transformers import Siglip2VisionModel
            return Siglip2VisionModel.from_pretrained(self._MODEL_ID)
        except (ImportError, RuntimeError):
            from transformers import SiglipVisionModel
            return SiglipVisionModel.from_pretrained(self._MODEL_ID)

    def _embed(self, outputs):
        return outputs.pooler_output


class Siglip2LargeExtractor(Siglip2Extractor):
    name = "siglip2_large"
    _MODEL_ID = "google/siglip2-large-patch16-256"


class VitExtractor(_HFExtractor):
    name = "vit_base"
    _MODEL_ID = "google/vit-base-patch16-224"

    def _embed(self, outputs):
        return outputs.last_hidden_state[:, 0]


class Aimv2LargeExtractor(_HFExtractor):
    name = "aimv2_large"
    _MODEL_ID = "apple/aimv2-large-patch14-224"

    def _embed(self, outputs):
        # AIMv2 is autoregressive with no CLS token; mean-pool patch tokens.
        return outputs.last_hidden_state.mean(dim=1)
