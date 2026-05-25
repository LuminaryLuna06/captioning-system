"""Unit tests for the retrieval closures. Uses an in-memory tiny FAISS index."""
from __future__ import annotations

import numpy as np
import faiss
from PIL import Image

from hanoi_caption.retrieval.retrieve import make_retrieve_fn, make_topk_fn


class FakeExtractor:
    name = "fake"
    dim = 3

    def __init__(self, fixed_emb=None):
        self.fixed_emb = fixed_emb

    def extract(self, images):
        if self.fixed_emb is not None:
            return np.tile(self.fixed_emb, (len(images), 1)).astype("float32")
        return np.ones((len(images), self.dim), dtype="float32") / np.sqrt(self.dim)


def _build_index(vectors, paths):
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(vectors.astype("float32"))
    id_map = {i: p for i, p in enumerate(paths)}
    return idx, id_map


def test_make_retrieve_fn_returns_kb_id():
    v_a = np.array([[1.0, 0.0, 0.0]])
    v_b = np.array([[0.0, 1.0, 0.0]])
    index, id_map = _build_index(
        np.vstack([v_a, v_b]),
        ["data/kb_images/kb_alpha/img1.jpg", "data/kb_images/kb_beta/img2.jpg"],
    )
    ext = FakeExtractor(fixed_emb=np.array([1.0, 0.0, 0.0]))
    retrieve = make_retrieve_fn(ext, index, id_map)
    kb_id, score = retrieve(Image.new("RGB", (4, 4)))
    assert kb_id == "kb_alpha"
    assert score == 1.0


def test_returns_none_when_index_empty():
    empty_index = faiss.IndexFlatIP(3)
    ext = FakeExtractor()
    retrieve = make_retrieve_fn(ext, empty_index, {})
    kb_id, score = retrieve(Image.new("RGB", (4, 4)))
    assert kb_id is None
    assert score == 0.0


def test_make_topk_fn_returns_k_results():
    v_a = np.array([[1.0, 0.0, 0.0]])
    v_b = np.array([[0.0, 1.0, 0.0]])
    v_c = np.array([[0.0, 0.0, 1.0]])
    index, id_map = _build_index(
        np.vstack([v_a, v_b, v_c]),
        [
            "data/kb_images/kb_alpha/img1.jpg",
            "data/kb_images/kb_beta/img2.jpg",
            "data/kb_images/kb_gamma/img3.jpg",
        ],
    )
    ext = FakeExtractor(fixed_emb=np.array([1.0, 0.0, 0.0]))
    topk = make_topk_fn(ext, index, id_map, k=3)
    results = topk(Image.new("RGB", (4, 4)))
    assert len(results) == 3
    assert results[0]["kb_id"] == "kb_alpha"
    assert results[0]["score"] == 1.0
    for r in results:
        assert {"path", "kb_id", "score"} <= set(r.keys())


def test_make_topk_fn_handles_missing_path():
    v_a = np.array([[1.0, 0.0, 0.0]])
    v_b = np.array([[0.0, 1.0, 0.0]])
    index = faiss.IndexFlatIP(3)
    index.add(np.vstack([v_a, v_b]).astype("float32"))
    # id_map is missing key 1 — simulate a corrupt or out-of-sync mapping
    id_map = {0: "data/kb_images/kb_alpha/img1.jpg"}
    ext = FakeExtractor(fixed_emb=np.array([0.0, 1.0, 0.0]))   # query closest to v_b at idx 1
    topk = make_topk_fn(ext, index, id_map, k=2)
    results = topk(Image.new("RGB", (4, 4)))
    assert len(results) == 2
    assert results[0]["kb_id"] is None       # idx 1 has no path → kb_id None
    assert results[0]["path"] is None        # consistent sentinel
    assert results[1]["kb_id"] == "kb_alpha" # idx 0 still works
