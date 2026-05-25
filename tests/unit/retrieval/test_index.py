"""Unit tests for the FAISS cache builder/loader. Uses a deterministic fake
extractor so these run < 1s on CPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hanoi_caption.retrieval.index import build_or_load_index


class FakeExtractor:
    name = "fake"
    dim = 4

    def __init__(self):
        self.calls = 0

    def extract(self, images):
        self.calls += 1
        out = []
        for img in images:
            buf = np.array(img.resize((4, 4))).tobytes()
            h = hashlib.sha256(buf).digest()
            vec = np.frombuffer(h[: 4 * 4], dtype=np.uint8).astype(np.float32).reshape(4, 4).mean(0)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            out.append(vec)
        return np.stack(out).astype(np.float32)


def _make_kb(tmp: Path):
    """Create tmp/kb_images/<kb>/<file>.png for two landmarks."""
    for kb, fname, color in [
        ("kb_a", "img1.png", (255, 0, 0)),
        ("kb_a", "img2.png", (0, 255, 0)),
        ("kb_b", "img3.png", (0, 0, 255)),
    ]:
        d = tmp / "kb_images" / kb
        d.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), color).save(d / fname)
    return tmp / "kb_images"


def test_build_creates_cache(tmp_path):
    kb_dir = _make_kb(tmp_path)
    cache_dir = tmp_path / "cache"

    index, id_map = build_or_load_index(FakeExtractor(), kb_dir, cache_dir)

    assert (cache_dir / "fake" / "faiss.index").exists()
    assert (cache_dir / "fake" / "id_map.json").exists()
    assert index.ntotal == 3
    assert set(id_map.keys()) == {0, 1, 2}
    assert all(str(kb_dir) in v for v in id_map.values())


def test_load_skips_rebuild(tmp_path):
    kb_dir = _make_kb(tmp_path)
    cache_dir = tmp_path / "cache"

    build_or_load_index(FakeExtractor(), kb_dir, cache_dir)

    ext2 = FakeExtractor()
    build_or_load_index(ext2, kb_dir, cache_dir)
    assert ext2.calls == 0, "second call should hit cache, not re-extract"


def test_force_rebuild_calls_extract(tmp_path):
    kb_dir = _make_kb(tmp_path)
    cache_dir = tmp_path / "cache"

    build_or_load_index(FakeExtractor(), kb_dir, cache_dir)

    ext2 = FakeExtractor()
    build_or_load_index(ext2, kb_dir, cache_dir, force_rebuild=True)
    assert ext2.calls > 0


def test_search_returns_nearest(tmp_path):
    kb_dir = _make_kb(tmp_path)
    cache_dir = tmp_path / "cache"

    ext = FakeExtractor()
    index, id_map = build_or_load_index(ext, kb_dir, cache_dir)

    query_path = kb_dir / "kb_a" / "img1.png"
    query_emb = ext.extract([Image.open(query_path)])
    scores, indices = index.search(query_emb, k=1)

    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)
    assert "img1.png" in id_map[int(indices[0][0])]


def test_empty_kb_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty_kb"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="no images"):
        build_or_load_index(FakeExtractor(), empty_dir, tmp_path / "cache")
