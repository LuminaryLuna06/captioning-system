"""Embed KB Visual Cues and build an in-memory cosine index."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import KBNode, MatchCandidate

CACHE_DIR = Path("data/cache")
EMBEDDING_MODEL_NAME = "bge_m3"


def _load_bge_m3():
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


registry.register(EMBEDDING_MODEL_NAME, _load_bge_m3)


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def embed_text(texts: list[str]) -> np.ndarray:
    model = registry.get(EMBEDDING_MODEL_NAME)
    out = model.encode(texts, batch_size=8, max_length=1024)["dense_vecs"]
    arr = np.asarray(out, dtype=np.float32)
    return _normalize(arr)


@dataclass
class KBIndex:
    node_ids: list[str]
    embeddings: np.ndarray  # shape (N, D), L2-normalized

    def topk(self, query: np.ndarray, k: int) -> list[MatchCandidate]:
        q = _normalize(query.reshape(1, -1))[0]
        sims = self.embeddings @ q  # (N,)
        idx = np.argsort(-sims)[:k]
        return [
            MatchCandidate(node_id=self.node_ids[i], score=float(sims[i]))
            for i in idx
        ]


def _kb_hash(nodes: dict[str, KBNode]) -> str:
    payload = json.dumps(
        {nid: n.visual_cues_en for nid, n in sorted(nodes.items())},
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def build_or_load_index(nodes: dict[str, KBNode]) -> KBIndex:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = _kb_hash(nodes)
    cache_path = CACHE_DIR / f"kb_index_{h}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        ids = list(data["node_ids"])
        embs = data["embeddings"].astype(np.float32)
        return KBIndex(node_ids=ids, embeddings=embs)

    ids = list(nodes.keys())
    cues = [nodes[nid].visual_cues_en for nid in ids]
    embs = embed_text(cues)
    np.savez(cache_path, node_ids=np.array(ids), embeddings=embs)
    return KBIndex(node_ids=ids, embeddings=embs)
