# Retriever Backbone Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plug-and-play notebook that loads four image-embedding backbones (DINOv3, ResNet-50, SigLIP-2, ViT) and qualitatively compares their per-frame retrieval against the landmark KB — without touching the production pipeline.

**Architecture:** A new self-contained module `hanoi_caption/retrieval/` (Protocol + concrete extractors + FAISS cache + retrieve closures), a one-time CLI to populate per-backbone caches under `data/cache/<name>/`, a one-time helper to extract fixed query frames into `tests/fixtures/retriever_frames/`, and a thin notebook `notebooks/03_retriever_comparison.ipynb` that orchestrates everything.

**Tech Stack:** Python 3.11, PyTorch + CUDA 12.8 (luna_env), `transformers`, FAISS-CPU (already installed), pytest 8, matplotlib, OpenCV (for video frame decode via existing `sample_frames`).

**Spec:** `docs/superpowers/specs/2026-05-25-retriever-backbone-comparison-design.md`

**Environment reminder:** All Python commands use `D:\Jupiter\luna_env\Scripts\python.exe`. From PowerShell on Windows, run `& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest ...`. The repo root is `D:\Jupiter\captioning-system`; all paths below are relative to that root unless absolute.

---

## Task 1: Module skeleton + protocol + slow marker registration

**Files:**
- Create: `hanoi_caption/retrieval/__init__.py`
- Create: `hanoi_caption/retrieval/backbones.py`
- Modify: `pyproject.toml` (add `markers = ["slow: ..."]` under `[tool.pytest.ini_options]`)

- [ ] **Step 1: Create the package directory and empty `__init__.py`**

```bash
mkdir -p hanoi_caption/retrieval
```

Write `hanoi_caption/retrieval/__init__.py` (empty file is fine).

- [ ] **Step 2: Write the `BackboneExtractor` protocol**

Write `hanoi_caption/retrieval/backbones.py`:

```python
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
```

(Base class and concrete extractors are added in Task 2.)

- [ ] **Step 3: Register the `slow` pytest marker**

Modify `pyproject.toml` — find the `[tool.pytest.ini_options]` section and add a `markers` key. After the change the section reads:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
markers = [
    "slow: tests that load real models or otherwise need GPU/network; skipped by default in CI via -m 'not slow'",
]
```

- [ ] **Step 4: Verify the protocol module imports cleanly**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -c "from hanoi_caption.retrieval.backbones import BackboneExtractor; print(BackboneExtractor)"
```
Expected: `<class 'hanoi_caption.retrieval.backbones.BackboneExtractor'>` (no traceback).

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/retrieval/__init__.py hanoi_caption/retrieval/backbones.py pyproject.toml
git commit -m "feat(retrieval): add module skeleton + BackboneExtractor protocol + slow marker"
```

---

## Task 2: `_HFExtractor` base + 4 concrete extractors + parametrized slow test

**Files:**
- Modify: `hanoi_caption/retrieval/backbones.py` (append base + 4 concrete classes)
- Create: `tests/models/test_backbones.py`

- [ ] **Step 1: Write the failing parametrized slow test**

Write `tests/models/test_backbones.py`:

```python
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

    # eval() mode -> deterministic for the same input
    feat2 = ext.extract([img, img])
    np.testing.assert_allclose(feat, feat2, atol=1e-5)
```

- [ ] **Step 2: Run the test, confirm it fails**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/models/test_backbones.py -m slow --collect-only
```
Expected: collection error / `ImportError: cannot import name 'Dinov3Extractor'` — proves the test will fail until classes exist.

- [ ] **Step 3: Implement `_HFExtractor` base + 4 concrete classes**

Append to `hanoi_caption/retrieval/backbones.py`:

```python
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


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


class Resnet50Extractor(_HFExtractor):
    name = "resnet50"
    _MODEL_ID = "microsoft/resnet-50"

    def _embed(self, outputs):
        # HF ResNet pooler_output shape: (B, C, 1, 1) -> flatten to (B, C)
        return outputs.pooler_output.flatten(1)


class Siglip2Extractor(_HFExtractor):
    name = "siglip2_base"
    _MODEL_ID = "google/siglip2-base-patch16-224"

    def _load_model(self):
        # Avoid allocating the text tower: pull only the vision module.
        try:
            from transformers import Siglip2VisionModel
            return Siglip2VisionModel.from_pretrained(self._MODEL_ID)
        except ImportError:
            from transformers import SiglipVisionModel
            return SiglipVisionModel.from_pretrained(self._MODEL_ID)

    def _embed(self, outputs):
        return outputs.pooler_output


class VitExtractor(_HFExtractor):
    name = "vit_base"
    _MODEL_ID = "google/vit-base-patch16-224"

    def _embed(self, outputs):
        return outputs.last_hidden_state[:, 0]
```

- [ ] **Step 4: Run the slow test, confirm it passes**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/models/test_backbones.py -m slow -v
```
Expected: 4 passed. First run downloads weights (~1-2 minutes); subsequent runs are fast (~10-20 s for all four).

If both `Siglip2VisionModel` and `SiglipVisionModel` fail to load `siglip2-base-patch16-224`, upgrade `transformers` (`& "D:\Jupiter\luna_env\Scripts\python.exe" -m pip install -U transformers`) — do not add a third silent fallback to `AutoModel`, as loading the text tower defeats the design.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/retrieval/backbones.py tests/models/test_backbones.py
git commit -m "feat(retrieval): implement DINOv3/ResNet-50/SigLIP-2/ViT extractors with parametrized slow test"
```

---

## Task 3: `build_or_load_index` with fake-extractor unit tests

**Files:**
- Create: `hanoi_caption/retrieval/index.py`
- Create: `tests/unit/retrieval/__init__.py` (empty; pytest discovers but stays tidy)
- Create: `tests/unit/retrieval/test_index.py`

- [ ] **Step 1: Write the failing unit tests**

Write `tests/unit/retrieval/__init__.py` (empty).

Write `tests/unit/retrieval/test_index.py`:

```python
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
        # Hash filenames isn't possible from PIL alone; hash the bytes for determinism.
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
    # id_map values are absolute paths under kb_dir
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

    # Re-embed the first KB image as a query
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
```

- [ ] **Step 2: Run tests, confirm they fail**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/unit/retrieval/test_index.py -v
```
Expected: `ModuleNotFoundError: hanoi_caption.retrieval.index` — proves the tests need the implementation.

- [ ] **Step 3: Implement `build_or_load_index`**

Write `hanoi_caption/retrieval/index.py`:

```python
"""Build-or-load FAISS index per backbone, cached under {cache_dir}/{extractor.name}/."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

_VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(kb_dir: Path) -> list[Path]:
    return sorted(p for p in kb_dir.rglob("*") if p.suffix.lower() in _VALID_EXT)


def build_or_load_index(
    extractor,
    kb_images_dir: Path | str,
    cache_dir: Path | str = "data/cache",
    batch_size: int = 16,
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, dict[int, str]]:
    """Return (faiss_index, id_map). Build if cache missing, else load."""
    kb_dir = Path(kb_images_dir)
    cache_path = Path(cache_dir) / extractor.name
    index_path = cache_path / "faiss.index"
    map_path = cache_path / "id_map.json"

    if not force_rebuild and index_path.exists() and map_path.exists():
        log.info("Loading cached index for %s from %s", extractor.name, cache_path)
        index = faiss.read_index(str(index_path))
        with open(map_path, "r", encoding="utf-8") as f:
            id_map = {int(k): v for k, v in json.load(f).items()}
        return index, id_map

    image_paths = _list_images(kb_dir)
    if not image_paths:
        raise ValueError(f"no images found under {kb_dir}")

    cache_path.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(extractor.dim)
    id_map: dict[int, str] = {}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        valid_paths = []
        for p in batch_paths:
            try:
                batch_images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                log.warning("skipping unreadable image %s: %s", p, e)
        if not batch_images:
            continue
        embeddings = extractor.extract(batch_images).astype("float32")
        start_id = index.ntotal
        for j, p in enumerate(valid_paths):
            id_map[start_id + j] = str(p)
        index.add(embeddings)
        log.info("indexed %d/%d images for %s", index.ntotal, len(image_paths), extractor.name)

    faiss.write_index(index, str(index_path))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
    return index, id_map
```

- [ ] **Step 4: Run tests, confirm they pass**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/unit/retrieval/test_index.py -v
```
Expected: 5 passed in < 2s.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/retrieval/index.py tests/unit/retrieval/__init__.py tests/unit/retrieval/test_index.py
git commit -m "feat(retrieval): add build_or_load_index with per-backbone FAISS cache"
```

---

## Task 4: `make_retrieve_fn` + `make_topk_fn` with unit tests

**Files:**
- Create: `hanoi_caption/retrieval/retrieve.py`
- Create: `tests/unit/retrieval/test_retrieve.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/retrieval/test_retrieve.py`:

```python
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
        # Default: deterministic per call
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
    # Query identical to v_a
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
```

- [ ] **Step 2: Run tests, confirm they fail**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/unit/retrieval/test_retrieve.py -v
```
Expected: `ModuleNotFoundError: hanoi_caption.retrieval.retrieve`.

- [ ] **Step 3: Implement the retrieve closures**

Write `hanoi_caption/retrieval/retrieve.py`:

```python
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
```

- [ ] **Step 4: Run tests, confirm they pass**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/unit/retrieval/test_retrieve.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/retrieval/retrieve.py tests/unit/retrieval/test_retrieve.py
git commit -m "feat(retrieval): add make_retrieve_fn and make_topk_fn closures"
```

---

## Task 5: CLI `build_all_backbones.py`

**Files:**
- Create: `scripts/data_collection/build_all_backbones.py`

- [ ] **Step 1: Write the CLI**

Write `scripts/data_collection/build_all_backbones.py`:

```python
"""Build the FAISS retrieval cache once per backbone.

    python scripts/data_collection/build_all_backbones.py
        [--backbones dinov3,resnet50,siglip2,vit]   # default: all four
        [--kb-dir data/kb_images]
        [--cache-dir data/cache]
        [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as a script without `pip install -e .`
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hanoi_caption.retrieval.backbones import (  # noqa: E402
    Dinov3Extractor,
    Resnet50Extractor,
    Siglip2Extractor,
    VitExtractor,
)
from hanoi_caption.retrieval.index import build_or_load_index  # noqa: E402

REGISTRY = {
    "dinov3":   Dinov3Extractor,
    "resnet50": Resnet50Extractor,
    "siglip2":  Siglip2Extractor,
    "vit":      VitExtractor,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", default=",".join(REGISTRY.keys()),
                   help="Comma-separated subset of: " + ",".join(REGISTRY.keys()))
    p.add_argument("--kb-dir", default="data/kb_images", type=Path)
    p.add_argument("--cache-dir", default="data/cache", type=Path)
    p.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("build_all_backbones")

    requested = [b.strip() for b in args.backbones.split(",") if b.strip()]
    unknown = [b for b in requested if b not in REGISTRY]
    if unknown:
        log.error("unknown backbones: %s (valid: %s)", unknown, list(REGISTRY))
        return 2

    for name in requested:
        log.info("=== %s ===", name)
        t0 = time.perf_counter()
        ext = REGISTRY[name]()
        index, id_map = build_or_load_index(
            ext, args.kb_dir, args.cache_dir, force_rebuild=args.force,
        )
        log.info("%s ready: %d vectors, dim=%d, %.1fs",
                 name, index.ntotal, ext.dim, time.perf_counter() - t0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke test on a tiny fake KB dir**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" scripts/data_collection/build_all_backbones.py --help
```
Expected: argparse usage text, exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/data_collection/build_all_backbones.py
git commit -m "feat(scripts): CLI to build per-backbone FAISS caches"
```

---

## Task 6: One-shot helper `extract_fixed_frames.py`

**Files:**
- Create: `scripts/data_collection/extract_fixed_frames.py`

- [ ] **Step 1: Write the helper**

Write `scripts/data_collection/extract_fixed_frames.py`:

```python
"""Extract one representative frame per landmark into tests/fixtures/retriever_frames/
for use as fixed query images by notebooks/03_retriever_comparison.ipynb.

Run once after pulling videos into tests/videos/:

    python scripts/data_collection/extract_fixed_frames.py
        [--video-dir tests/videos]
        [--out-dir tests/fixtures/retriever_frames]

Hard-coded (video_filename, timestamp_s, kb_id) triples — edit the list below
to add or remove landmarks.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# (video filename under --video-dir, timestamp in seconds, kb_id to save as).
# The kb_id strings are only used as the saved filename + the "expected" label
# in the notebook's top-K grid; they DO NOT need to match real kb_ids in
# data/kb.json. Update them to match if you want the grid title to read as a
# proper kb_id (run `ls data/kb_images/` to see the canonical names).
FIXED_FRAMES = [
    ("NhaThoLon_S_T03.MOV",                  10.0, "nha_tho_lon"),
    ("NhaHatLon_S_T04.MOV",                  10.0, "nha_hat_lon"),
    ("NhaKhachChinhPhu_S_T02.MOV",           10.0, "nha_khach_chinh_phu"),
    ("A1_018_DenNgocSonToanCanh_M_T02.mp4",  10.0, "den_ngoc_son"),
    ("FLN_BaoTangGom_T48_S.MOV",             10.0, "bao_tang_gom"),
]


def extract_one(video_path: Path, timestamp_s: float):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"unreadable FPS for {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(timestamp_s * fps)))
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame at t={timestamp_s}s from {video_path}")
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video-dir", default="tests/videos", type=Path)
    p.add_argument("--out-dir", default="tests/fixtures/retriever_frames", type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("extract_fixed_frames")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for filename, t, kb_id in FIXED_FRAMES:
        video_path = args.video_dir / filename
        if not video_path.exists():
            log.warning("missing: %s", video_path)
            continue
        try:
            img = extract_one(video_path, t)
        except Exception as e:
            log.warning("failed %s (%s): %s", filename, kb_id, e)
            continue
        out_path = args.out_dir / f"{kb_id}.jpg"
        img.save(out_path, "JPEG", quality=92)
        log.info("wrote %s", out_path)
        n_ok += 1

    log.info("extracted %d/%d fixed frames", n_ok, len(FIXED_FRAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit (without running yet)**

```bash
git add scripts/data_collection/extract_fixed_frames.py
git commit -m "feat(scripts): one-shot helper to extract fixed query frames for notebook"
```

---

## Task 7: Populate cache + fixtures (manual, no commit)

This task only runs scripts to fill local-only directories (`data/cache/`, `tests/fixtures/`) which are gitignored. No code changes, no commit.

- [ ] **Step 1: Build all four backbone caches**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" scripts/data_collection/build_all_backbones.py
```
Expected output (per backbone): `=== <name> ===`, `indexed N/M images for <name>`, `<name> ready: N vectors, dim=D, T.Ts`. Four sections total. First run downloads weights; subsequent runs only extract (~seconds per backbone).

- [ ] **Step 2: Verify each cache exists**

Run:
```
ls data/cache/dinov3_vits16/ data/cache/resnet50/ data/cache/siglip2_base/ data/cache/vit_base/
```
Expected: `faiss.index` + `id_map.json` in each.

- [ ] **Step 3: Extract fixed frames**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" scripts/data_collection/extract_fixed_frames.py
```
Expected output: `wrote tests/fixtures/retriever_frames/<kb_id>.jpg` for each of the five triples.

- [ ] **Step 4: Verify fixtures exist**

Run:
```
ls tests/fixtures/retriever_frames/
```
Expected: 5 JPGs named after kb_ids (`nha_tho_lon.jpg`, `nha_hat_lon.jpg`, `nha_khach_chinh_phu.jpg`, `den_ngoc_son.jpg`, `bao_tang_gom.jpg`). If any are missing, check whether the source video is present under `tests/videos/`.

(No commit — both directories are gitignored.)

---

## Task 8: Notebook cells 1-3 (config + load + viz helpers)

**Files:**
- Create: `notebooks/03_retriever_comparison.ipynb`

The notebook is created via Jupyter or the `nbformat` API. Easiest path: build it programmatically with a tiny Python script to guarantee exact cell contents and avoid hand-editing JSON.

- [ ] **Step 1: Generate the notebook with the first three cells**

Run this Python snippet (or paste into a scratch `.py` and execute):

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = [
    nbf.v4.new_markdown_cell(
        "# Retriever backbone comparison\n\n"
        "Qualitative side-by-side of DINOv3 / ResNet-50 / SigLIP-2 / ViT as per-frame retrievers.\n\n"
        "**Prereqs (run once):**\n"
        "1. `python scripts/data_collection/build_all_backbones.py` — builds FAISS caches.\n"
        "2. `python scripts/data_collection/extract_fixed_frames.py` — extracts fixed query frames.\n\n"
        "See `docs/superpowers/specs/2026-05-25-retriever-backbone-comparison-design.md` for design context."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path.cwd().parent))   # repo root\n"
        "\n"
        "from PIL import Image\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from hanoi_caption.retrieval.backbones import (\n"
        "    Dinov3Extractor, Resnet50Extractor, Siglip2Extractor, VitExtractor,\n"
        ")\n"
        "from hanoi_caption.retrieval.index import build_or_load_index\n"
        "from hanoi_caption.retrieval.retrieve import make_retrieve_fn, make_topk_fn\n"
        "from hanoi_caption.video_pipeline import sample_frames\n"
        "\n"
        "KB_DIR         = Path('../data/kb_images')\n"
        "CACHE_DIR      = Path('../data/cache')\n"
        "FIXED_FRAMES   = Path('../tests/fixtures/retriever_frames')\n"
        "TIMELINE_VIDEO = Path('../tests/videos/NhaThoLon_S_T03.MOV')\n"
        "TOPK = 5\n"
        "SAMPLE_FPS = 1.0\n"
    ),
    nbf.v4.new_code_cell(
        "EXTRACTORS = {\n"
        "    'dinov3':   Dinov3Extractor(),\n"
        "    'resnet50': Resnet50Extractor(),\n"
        "    'siglip2':  Siglip2Extractor(),\n"
        "    'vit':      VitExtractor(),\n"
        "}\n"
        "INDEXES = {n: build_or_load_index(e, KB_DIR, CACHE_DIR) for n, e in EXTRACTORS.items()}\n"
        "for n, (idx, _) in INDEXES.items():\n"
        "    print(f'{n:10s} dim={EXTRACTORS[n].dim:4d}  vectors={idx.ntotal}')\n"
    ),
    nbf.v4.new_code_cell(
        "# Section A helper: top-K grid\n"
        "def show_topk_grid(queries, topk_by_model, cell_size=(2.0, 2.0)):\n"
        "    model_names = list(topk_by_model.keys())\n"
        "    k = len(next(iter(topk_by_model.values()))[0])\n"
        "    n_rows = len(queries)\n"
        "    n_cols = 1 + len(model_names) * k\n"
        "    fig, axes = plt.subplots(\n"
        "        n_rows, n_cols,\n"
        "        figsize=(cell_size[0] * n_cols, cell_size[1] * n_rows),\n"
        "    )\n"
        "    if n_rows == 1:\n"
        "        axes = axes[None, :]\n"
        "    for r, (label, qimg) in enumerate(queries):\n"
        "        axes[r, 0].imshow(qimg)\n"
        "        axes[r, 0].set_title(f'Q: {label}', fontsize=8)\n"
        "        axes[r, 0].axis('off')\n"
        "        c = 1\n"
        "        for mname in model_names:\n"
        "            for rank, res in enumerate(topk_by_model[mname][r]):\n"
        "                ax = axes[r, c]\n"
        "                ax.imshow(Image.open(res['path']))\n"
        "                if rank == 0:\n"
        "                    ax.set_title(f'{mname}\\n{res[\"kb_id\"]}\\n{res[\"score\"]:.2f}', fontsize=7)\n"
        "                else:\n"
        "                    ax.set_title(f'{res[\"kb_id\"]}\\n{res[\"score\"]:.2f}', fontsize=7)\n"
        "                ax.axis('off')\n"
        "                c += 1\n"
        "    plt.tight_layout()\n"
        "    return fig\n"
        "\n"
        "# Section B helper: timeline\n"
        "def show_timeline(timeline_by_model, sample_fps=1.0):\n"
        "    model_names = list(timeline_by_model.keys())\n"
        "    all_kbs = sorted({kb for tl in timeline_by_model.values() for _, kb, _ in tl if kb})\n"
        "    cmap = plt.cm.tab20\n"
        "    palette = {kb: list(cmap(i % 20)) for i, kb in enumerate(all_kbs)}\n"
        "    UNKNOWN = (0.85, 0.85, 0.85, 1.0)\n"
        "    stride = 1.0 / sample_fps\n"
        "    fig, ax = plt.subplots(figsize=(14, 1.0 * len(model_names) + 1))\n"
        "    for row, mname in enumerate(model_names):\n"
        "        for t, kb, score in timeline_by_model[mname]:\n"
        "            color = list(palette.get(kb, UNKNOWN))\n"
        "            color[3] = min(1.0, max(0.3, float(score)))\n"
        "            ax.barh(row, stride, left=t, height=0.8, color=color, edgecolor='none')\n"
        "    ax.set_yticks(range(len(model_names)))\n"
        "    ax.set_yticklabels(model_names)\n"
        "    ax.set_xlabel('time (s)')\n"
        "    handles = [plt.Line2D([0], [0], color=palette[k], lw=6, label=k) for k in all_kbs]\n"
        "    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)\n"
        "    plt.tight_layout()\n"
        "    return fig\n"
    ),
]

with open("notebooks/03_retriever_comparison.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote notebooks/03_retriever_comparison.ipynb")
```

- [ ] **Step 2: Verify the notebook opens cleanly**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -c "import nbformat; nb = nbformat.read('notebooks/03_retriever_comparison.ipynb', as_version=4); print(len(nb.cells), 'cells')"
```
Expected: `4 cells` (1 markdown + 3 code).

- [ ] **Step 3: Commit (notebook with helpers, before adding Sections A/B)**

```bash
git add notebooks/03_retriever_comparison.ipynb
git commit -m "feat(notebook): retriever-comparison notebook skeleton (config + load + viz helpers)"
```

---

## Task 9: Notebook cell 4 — Section A (top-K view)

**Files:**
- Modify: `notebooks/03_retriever_comparison.ipynb` (append one code cell)

- [ ] **Step 1: Append the Section A cell programmatically**

Run this snippet:

```python
import nbformat as nbf

path = "notebooks/03_retriever_comparison.ipynb"
nb = nbf.read(path, as_version=4)
nb.cells.append(nbf.v4.new_markdown_cell("## Section A — Top-K view\nFor each fixed query frame, show the top-5 nearest KB images per backbone."))
nb.cells.append(nbf.v4.new_code_cell(
    "fixed_paths = sorted(FIXED_FRAMES.glob('*.jpg'))\n"
    "assert fixed_paths, f'No fixtures at {FIXED_FRAMES} — run extract_fixed_frames.py first'\n"
    "queries = [(p.stem, Image.open(p)) for p in fixed_paths]\n"
    "\n"
    "topk_results = {}\n"
    "for name, ext in EXTRACTORS.items():\n"
    "    fn = make_topk_fn(ext, *INDEXES[name], k=TOPK)\n"
    "    topk_results[name] = [fn(img) for _, img in queries]\n"
    "\n"
    "_ = show_topk_grid(queries, topk_results)\n"
))
nbf.write(nb, open(path, "w", encoding="utf-8"))
print("appended Section A cells")
```

- [ ] **Step 2: Open the notebook in Jupyter and run all cells**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m jupyter lab notebooks/03_retriever_comparison.ipynb
```

In Jupyter, run all cells. Expected result: a grid figure with 5 rows (one per query) and `1 + 4*5 = 21` columns (query + top-5 per model). Visually verify that DINOv3's rank-1 match for each query is the same `kb_id` as the query stem in most cases.

- [ ] **Step 3: Commit (with cleared outputs)**

Clear notebook outputs before committing:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m jupyter nbconvert --clear-output --inplace notebooks/03_retriever_comparison.ipynb
```

Then:
```bash
git add notebooks/03_retriever_comparison.ipynb
git commit -m "feat(notebook): Section A — top-K view per backbone on fixed query frames"
```

---

## Task 10: Notebook cell 5 — Section B (timeline view)

**Files:**
- Modify: `notebooks/03_retriever_comparison.ipynb` (append one code cell)

- [ ] **Step 1: Append the Section B cell programmatically**

Run:

```python
import nbformat as nbf

path = "notebooks/03_retriever_comparison.ipynb"
nb = nbf.read(path, as_version=4)
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section B — Timeline view\n"
    "Sample `TIMELINE_VIDEO` at 1 fps and plot per-frame predicted `kb_id` (colored) "
    "with score as opacity, one band per backbone."
))
nb.cells.append(nbf.v4.new_code_cell(
    "assert TIMELINE_VIDEO.exists(), f'Missing {TIMELINE_VIDEO}'\n"
    "sampled = sample_frames(TIMELINE_VIDEO, sample_fps=SAMPLE_FPS)\n"
    "frames = [img for _, _, img in sampled]\n"
    "times  = [t   for _, t, _   in sampled]\n"
    "print(f'sampled {len(frames)} frames from {TIMELINE_VIDEO.name}')\n"
    "\n"
    "timeline_by_model = {}\n"
    "for name, ext in EXTRACTORS.items():\n"
    "    fn = make_retrieve_fn(ext, *INDEXES[name])\n"
    "    timeline_by_model[name] = [(t, *fn(img)) for img, t in zip(frames, times)]\n"
    "\n"
    "_ = show_timeline(timeline_by_model, sample_fps=SAMPLE_FPS)\n"
))
nbf.write(nb, open(path, "w", encoding="utf-8"))
print("appended Section B cells")
```

- [ ] **Step 2: Run the new cell in Jupyter**

Reload the notebook in Jupyter (`File > Reload Notebook from Disk`) and run the new Section B cells. Expected: console print "sampled N frames from NhaThoLon_S_T03.MOV", then a stacked-band figure with four rows (dinov3 / resnet50 / siglip2 / vit), x = seconds. The Nha Tho Lon band should dominate for at least one backbone.

- [ ] **Step 3: Commit (with cleared outputs)**

```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m jupyter nbconvert --clear-output --inplace notebooks/03_retriever_comparison.ipynb
git add notebooks/03_retriever_comparison.ipynb
git commit -m "feat(notebook): Section B — timeline view of per-frame retrieval per backbone"
```

---

## Task 11: Notebook cell 6 — quick stats + final cleanup

**Files:**
- Modify: `notebooks/03_retriever_comparison.ipynb` (append one code cell)

- [ ] **Step 1: Append the stats cell programmatically**

Run:

```python
import nbformat as nbf

path = "notebooks/03_retriever_comparison.ipynb"
nb = nbf.read(path, as_version=4)
nb.cells.append(nbf.v4.new_markdown_cell(
    "## Section C — Quick stats\n"
    "Coarse signal until ground-truth segmentation lands. Latency is per-frame on CUDA, "
    "averaged over the timeline frames; `frac score > 0.5` is per-backbone (cosine scale "
    "is *not* strictly cross-backbone comparable)."
))
nb.cells.append(nbf.v4.new_code_cell(
    "import time\n"
    "\n"
    "rows = []\n"
    "for name, ext in EXTRACTORS.items():\n"
    "    fn = make_retrieve_fn(ext, *INDEXES[name])\n"
    "    # warm-up\n"
    "    fn(frames[0])\n"
    "    t0 = time.perf_counter()\n"
    "    preds = [fn(img) for img in frames]\n"
    "    dt = (time.perf_counter() - t0) / len(frames) * 1000\n"
    "    high = sum(1 for kb, sc in preds if sc > 0.5)\n"
    "    rows.append((name, ext.dim, f'{dt:6.1f} ms', f'{high/len(preds):.0%}'))\n"
    "\n"
    "print(f'{\"model\":10s} {\"dim\":>5s} {\"latency/frame\":>14s} {\"frac score>0.5\":>16s}')\n"
    "print('-' * 50)\n"
    "for r in rows:\n"
    "    print(f'{r[0]:10s} {r[1]:5d} {r[2]:>14s} {r[3]:>16s}')\n"
))
nbf.write(nb, open(path, "w", encoding="utf-8"))
print("appended Section C cells")
```

- [ ] **Step 2: Run the stats cell in Jupyter**

Reload, run. Expected: a 4-row text table with latency in ms (DINOv3 + ResNet typically < 20 ms/frame, ViT + SigLIP often 30-60 ms on a 16 GB RTX) and score-above-0.5 fraction.

- [ ] **Step 3: Final full-suite test run**

Run:
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/ -m "not slow" -v
```
Expected: all unit + integration tests pass (the new retrieval unit tests added, nothing else regressed).

Optionally run the slow tests (one-shot):
```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m pytest tests/models -m slow -v
```
Expected: 4 passed.

- [ ] **Step 4: Commit (with cleared outputs)**

```
& "D:\Jupiter\luna_env\Scripts\python.exe" -m jupyter nbconvert --clear-output --inplace notebooks/03_retriever_comparison.ipynb
git add notebooks/03_retriever_comparison.ipynb
git commit -m "feat(notebook): Section C — per-backbone latency + score-distribution stats"
```

- [ ] **Step 5: Document the new module in CLAUDE.md**

Add this short paragraph under the existing "Modules" section (between `hanoi_caption/schemas.py` and "Knowledge base"):

```markdown
- `hanoi_caption/retrieval/` — experimental pluggable backbones for retrieval comparison
  (`backbones.py`: DINOv3/ResNet-50/SigLIP-2/ViT extractors; `index.py`:
  `build_or_load_index` with per-backbone FAISS cache under `data/cache/<name>/`;
  `retrieve.py`: `make_retrieve_fn` / `make_topk_fn` closures). Production pipeline
  is unchanged and still uses `scripts/data_collection/FeatureExtractor` + `ImageIndexer`.
  Notebook: `notebooks/03_retriever_comparison.ipynb`.
```

And add this paragraph under "Data collection":

```markdown
**Per-backbone caches for the comparison notebook:**

```bash
python scripts/data_collection/build_all_backbones.py    # builds data/cache/<name>/
python scripts/data_collection/extract_fixed_frames.py   # writes tests/fixtures/retriever_frames/<kb>.jpg
```
```

Commit:
```bash
git add CLAUDE.md
git commit -m "docs(claude): document hanoi_caption/retrieval/ module + per-backbone cache scripts"
```

---

## Done criteria

When all 11 tasks are complete:
- `pytest tests/unit/retrieval -v` → all green, < 2 s
- `pytest tests/models -m slow -v` → 4 passed (on GPU)
- `pytest tests/ -m "not slow"` → all green (no regressions in existing tests)
- `data/cache/{dinov3_vits16,resnet50,siglip2_base,vit_base}/` each contain `faiss.index` + `id_map.json`
- `tests/fixtures/retriever_frames/` contains 5 JPGs
- `notebooks/03_retriever_comparison.ipynb` runs top-to-bottom in a fresh kernel and produces (1) a top-K grid, (2) a stacked-band timeline, (3) a latency + score table
- No changes to `hanoi_caption/video_pipeline.py`, `scripts/data_collection/feature_extractor.py`, `scripts/data_collection/indexer.py`
- CLAUDE.md updated with the new module + scripts
