from pathlib import Path

import numpy as np

from hanoi_caption.kb_indexer import KBIndex


def test_kbindex_cosine_topk_returns_in_descending_order():
    node_ids = ["a", "b", "c"]
    # Hand-crafted unit vectors
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]],
        dtype=np.float32,
    )
    idx = KBIndex(node_ids=node_ids, embeddings=embeddings)
    query = np.array([1.0, 0.0], dtype=np.float32)
    top = idx.topk(query, k=3)
    assert [c.node_id for c in top] == ["a", "c", "b"]
    assert top[0].score > top[1].score > top[2].score
