"""Closures over a (extractor, faiss.Index, id_map) triple, for use in the
notebook or as a drop-in `retrieve_fn` for `caption_video(retrieve_fn=...)`."""
from __future__ import annotations

import os
from typing import Callable


def _kb_id_from_path(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def make_retrieve_fn(extractor, index, id_map) -> Callable:
    """Return a callable: PIL.Image -> (kb_id | None, score: float).

    Mirrors the contract of `hanoi_caption.video_pipeline._default_retrieve_fn`.
    """
    def _retrieve(frame_pil):
        feat = extractor.extract([frame_pil]).astype("float32")
        scores, indices = index.search(feat, k=1)
        idx = int(indices[0][0])
        if idx < 0:
            return None, 0.0
        path = id_map.get(idx)
        if not path:
            return None, float(scores[0][0])
        return _kb_id_from_path(path), float(scores[0][0])
    return _retrieve


def make_topk_fn(extractor, index, id_map, k: int) -> Callable:
    """Return a callable: PIL.Image -> list[{path, kb_id, score}] of length <= k."""
    def _topk(frame_pil):
        feat = extractor.extract([frame_pil]).astype("float32")
        scores, indices = index.search(feat, k=k)
        out = []
        for score, idx in zip(scores[0], indices[0]):
            i = int(idx)
            if i < 0:
                continue
            path = id_map.get(i, "")
            out.append({
                "path": path,
                "kb_id": _kb_id_from_path(path) if path else None,
                "score": float(score),
            })
        return out
    return _topk
