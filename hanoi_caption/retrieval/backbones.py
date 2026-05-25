"""Pluggable image-embedding backbones for KB retrieval comparison.

Each concrete extractor implements the `BackboneExtractor` protocol:
- `name`: short id used as the cache subdirectory name
- `dim`: embedding dimension
- `extract(images)`: return float32 (N, dim) L2-normalized embeddings
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class BackboneExtractor(Protocol):
    name: str
    dim: int

    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return float32 array of shape (len(images), dim), L2-normalized along axis=1."""
        ...
