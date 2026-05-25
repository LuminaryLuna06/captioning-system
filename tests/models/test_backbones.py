"""Smoke contract tests for each BackboneExtractor — verifies shape, dtype,
L2-normalization, and determinism on a real model load. Marked @slow because
each test downloads weights (cached on first run) and runs a CUDA forward.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from hanoi_caption.retrieval.backbones import (
    Dinov3Extractor,
    Resnet50Extractor,
    Siglip2Extractor,
    VitExtractor,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "cls,expected_name,expected_dim",
    [
        (Dinov3Extractor,   "dinov3_vits16", 384),
        (Resnet50Extractor, "resnet50",      2048),
        (Siglip2Extractor,  "siglip2_base",  768),
        (VitExtractor,      "vit_base",      768),
    ],
)
def test_backbone_contract(cls, expected_name, expected_dim):
    ext = cls()
    assert ext.name == expected_name
    assert ext.dim == expected_dim

    img = Image.new("RGB", (224, 224), color="red")
    feat = ext.extract([img, img])

    assert feat.shape == (2, expected_dim)
    assert feat.dtype == np.float32

    norms = np.linalg.norm(feat, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-4)

    feat2 = ext.extract([img, img])
    np.testing.assert_allclose(feat, feat2, atol=1e-5)
