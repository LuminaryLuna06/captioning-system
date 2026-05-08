from PIL import Image
import numpy as np

from hanoi_caption.kb_indexer import KBIndex
from hanoi_caption.kb_matcher import match_kb


def _img() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(0, 0, 0))


def test_match_returns_top1_when_above_threshold():
    idx = KBIndex(
        node_ids=["a", "b"],
        embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    def fake_embed(text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def fake_rerank(image, candidate_ids, kb):
        return ("a", 0.9)

    res = match_kb(
        image=_img(),
        holistic_desc="placeholder",
        kb_index=idx,
        kb_nodes={},  # rerank stub doesn't read it
        threshold=0.45,
        embed_fn=fake_embed,
        rerank_fn=fake_rerank,
    )
    assert res.node_id == "a"
    assert res.confidence == 0.9


def test_match_refuses_below_threshold():
    idx = KBIndex(
        node_ids=["a"],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    def fake_embed(text: str) -> np.ndarray:
        return np.array([0.1, 0.99], dtype=np.float32)

    def fake_rerank(image, candidate_ids, kb):
        return ("a", 0.5)

    res = match_kb(
        image=_img(),
        holistic_desc="placeholder",
        kb_index=idx,
        kb_nodes={},
        threshold=0.45,
        embed_fn=fake_embed,
        rerank_fn=fake_rerank,
    )
    assert res.node_id is None


def test_match_refuses_when_rerank_says_none():
    idx = KBIndex(
        node_ids=["a"],
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    def fake_embed(text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def fake_rerank(image, candidate_ids, kb):
        return (None, 0.0)

    res = match_kb(
        image=_img(),
        holistic_desc="placeholder",
        kb_index=idx,
        kb_nodes={},
        threshold=0.45,
        embed_fn=fake_embed,
        rerank_fn=fake_rerank,
    )
    assert res.node_id is None
