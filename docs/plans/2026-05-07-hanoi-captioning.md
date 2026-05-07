# Hanoi Captioning System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a baseline Jupyter-notebook image captioning system for Hanoi landmarks that combines a manually-built bilingual KB with the Describe Anything Model (DAM) for region-grounded, tour-guide-style English captions.

**Architecture:** A KB-spine pipeline. The `Visual Cues` field of each KB node serves both as a retrieval signal (cosine match against a holistic image description) and as a source of detection queries (Grounding DINO → SAM 2 → DAM region descriptions). A composer LLM weaves KB facts and region details into a 150–300 word paragraph. Sequential model loading keeps peak VRAM under 14 GB on a 16 GB Blackwell GPU.

**Tech Stack:** Python 3.11, PyTorch (nightly cu128 for sm_120), Transformers, Pydantic 2, BGE-M3 embeddings, Qwen2.5-VL-7B-Instruct, Qwen2.5-7B-Instruct, Grounding DINO (HF), SAM 2 (Meta), DAM-3B (NVIDIA), JupyterLab.

**Spec:** [`docs/specs/2026-05-07-hanoi-captioning-design.md`](../specs/2026-05-07-hanoi-captioning-design.md)

---

## File Structure

```
CaptioningSystem/
├── pyproject.toml                  # deps + setuptools config
├── .gitignore
├── README.md
├── data/
│   ├── kb.json                     # 15-node sample KB (Gemini-generated, replaceable)
│   └── cache/                      # gitignored: embeddings + HF model cache
├── notebooks/
│   ├── 01_phase1_kb_only.ipynb     # Phase 1 smoke test (no DAM)
│   └── 02_phase2_full_pipeline.ipynb  # Phase 2 full pipeline + comparison
├── hanoi_caption/
│   ├── __init__.py
│   ├── schemas.py                  # Pydantic: KBNode, MatchResult, Region, RegionDescription, CaptionResult
│   ├── kb_loader.py                # JSON → dict[id, KBNode]
│   ├── kb_indexer.py               # BGE-M3 embedder + .npz cache
│   ├── model_registry.py           # lazy-load + sequential eviction
│   ├── image_describer.py          # Qwen2.5-VL-7B
│   ├── kb_matcher.py               # cosine + VLM re-rank
│   ├── query_extractor.py          # Qwen2.5-7B → JSON list of noun phrases
│   ├── region_proposer.py          # Grounding DINO + SAM 2
│   ├── region_describer.py         # DAM-3B
│   ├── composer.py                 # Qwen2.5-7B → final paragraph
│   └── pipeline.py                 # end-to-end orchestration
└── tests/
    ├── conftest.py                 # shared fixtures
    ├── fixtures/                   # 5–10 test images (user-provided later)
    ├── test_schemas.py
    ├── test_kb_loader.py
    ├── test_kb_indexer.py
    ├── test_kb_matcher.py
    ├── test_region_proposer.py     # IoU filter, fallback logic
    └── test_pipeline.py            # decision-gate logic with mocked modules
```

**Decomposition principle:** each module owns one stage; cross-module values are Pydantic schemas; `model_registry.py` is the only place that touches `transformers.from_pretrained`. Pure logic (schemas, IoU filter, threshold gate, KB loading) is tested with `pytest`. Model-touching code is smoke-tested in the notebooks — that is the v1 evaluation surface per the spec.

---

## Phase 0 — Project Scaffolding

### Task 0.1: Initialize project structure

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `hanoi_caption/__init__.py` (empty)
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `data/.gitkeep`, `data/cache/.gitkeep`, `notebooks/.gitkeep`, `tests/fixtures/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
cd "/Users/dinhtronganh/Documents/World Model/CaptioningSystem"
mkdir -p hanoi_caption tests/fixtures data/cache notebooks
touch hanoi_caption/__init__.py tests/__init__.py
touch data/.gitkeep data/cache/.gitkeep notebooks/.gitkeep tests/fixtures/.gitkeep
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "hanoi_caption"
version = "0.1.0"
description = "KB-grounded Hanoi landmark image captioning baseline"
requires-python = ">=3.10"
dependencies = [
    # Torch is installed separately from the nightly cu128 index; not pinned here.
    "transformers>=4.46",
    "accelerate>=1.0",
    "bitsandbytes>=0.45",
    "sentencepiece",
    "pillow",
    "numpy",
    "pydantic>=2",
    "huggingface_hub",
    "einops",
    "FlagEmbedding",
    "matplotlib",
    "jupyterlab",
    "ipykernel",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov"]
# Vision deps are installed manually after the torch nightly is in place
# (groundingdino-py, sam2, NVIDIA describe-anything from git).

[tool.setuptools.packages.find]
include = ["hanoi_caption*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 3: Write `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/

# Virtual envs
.venv/
venv/

# Notebook
.ipynb_checkpoints/

# Project artifacts
data/cache/
data/*.npz
tests/fixtures/*
!tests/fixtures/.gitkeep

# OS
.DS_Store

# IDE
.vscode/
.idea/
```

- [ ] **Step 4: Write `README.md`**

```markdown
# CaptioningSystem

Baseline KB-grounded image captioning for Hanoi landmarks.

See [`docs/specs/2026-05-07-hanoi-captioning-design.md`](docs/specs/2026-05-07-hanoi-captioning-design.md) for design.
See [`docs/plans/2026-05-07-hanoi-captioning.md`](docs/plans/2026-05-07-hanoi-captioning.md) for the implementation plan.

## Setup (RTX 5060 Ti / Blackwell)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -e ".[dev]"
pip install groundingdino-py sam2
pip install git+https://github.com/NVlabs/describe-anything.git
```

## Run

```bash
jupyter lab notebooks/01_phase1_kb_only.ipynb
```
```

- [ ] **Step 5: Write empty `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
```

- [ ] **Step 6: Initialize git and commit**

```bash
git init
git add .
git commit -m "chore: scaffold project structure and pyproject.toml"
```

Expected: `main (root-commit) <hash>] chore: scaffold project structure...`

---

### Task 0.2: Activate `luna_env` and install non-vision dependencies

**Files:** none — this configures the local environment.

The user already maintains a working environment named **`luna_env`** that has PyTorch + CUDA 12.8 (Blackwell sm_120) installed correctly. Do NOT create a new `.venv` — reuse `luna_env`.

- [ ] **Step 1: Activate `luna_env`**

```bash
conda activate luna_env   # or: mamba activate luna_env
python -V
which python
```

Expected: a Python ≥3.11 interpreter located inside the `luna_env` directory.

- [ ] **Step 2: Verify CUDA + Blackwell support BEFORE installing anything else**

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); print('compute', torch.cuda.get_device_capability(0))"
```

Expected output includes `cuda True`, device `NVIDIA GeForce RTX 5060 Ti`, compute `(12, 0)`. **If compute capability is not (12, 0) or torch reports `False` for CUDA, stop and ask the user — do not attempt to reinstall torch in `luna_env` without their permission**, as it is their working environment.

- [ ] **Step 3: Install project + dev deps in editable mode (into `luna_env`)**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 4: Smoke-test pytest**

```bash
pytest --collect-only
```

Expected: collects 0 tests, no errors.

---

### Task 0.3: Vendor the sample KB into the project

**Files:**
- Move: `/Users/dinhtronganh/hanoi_kb_sample.json` → `data/kb.json`

- [ ] **Step 1: Copy sample KB into project**

```bash
cp /Users/dinhtronganh/hanoi_kb_sample.json data/kb.json
ls -la data/kb.json
```

Expected: file present, ~75 KB.

- [ ] **Step 2: Verify it parses and has 15 nodes**

```bash
python -c "import json; d=json.load(open('data/kb.json')); print(len(d), 'nodes'); print([n['id'] for n in d])"
```

Expected: `15 nodes` and the list of landmark ids.

- [ ] **Step 3: Commit**

```bash
git add data/kb.json
git commit -m "data: add 15-node Hanoi landmark KB sample"
```

---

## Phase 1 — KB-only smoke test

Implements the identification + composition path. No DAM, no detector, no segmenter.

### Task 1.1: Pydantic schemas

**Files:**
- Create: `hanoi_caption/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError

from hanoi_caption.schemas import (
    KBNode,
    MatchResult,
    Region,
    RegionDescription,
    CaptionResult,
)


def test_kbnode_minimal_object():
    node = KBNode(
        id="temple_of_literature",
        name_en="Temple of Literature",
        name_vi="Văn Miếu – Quốc Tử Giám",
        type="object",
        parent_id="categoryHaNoi",
        description_en="...",
        description_vi="...",
        visual_cues_en="stone steles, tiered roof",
        visual_cues_vi="...",
        tags=[],
    )
    assert node.id == "temple_of_literature"
    assert node.type == "object"


def test_kbnode_rejects_unknown_type():
    with pytest.raises(ValidationError):
        KBNode(
            id="x", name_en="x", name_vi="x", type="bogus",
            parent_id=None, description_en="x", description_vi="x",
            visual_cues_en="x", visual_cues_vi="x", tags=[],
        )


def test_match_result_none_path():
    r = MatchResult(node_id=None, confidence=0.0, top_k=[])
    assert r.node_id is None


def test_caption_result_either_caption_or_refusal():
    ok = CaptionResult(caption="A long paragraph...", refusal=None, debug={})
    assert ok.caption is not None
    refused = CaptionResult(caption=None, refusal="Not recognized.", debug={})
    assert refused.refusal is not None
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_schemas.py -v
```

Expected: ImportError or ModuleNotFoundError on `hanoi_caption.schemas`.

- [ ] **Step 3: Implement `hanoi_caption/schemas.py`**

```python
"""Pydantic schemas exchanged across pipeline modules."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class KBNode(BaseModel):
    id: str
    name_en: str
    name_vi: str
    type: Literal["object", "category"]
    parent_id: str | None = None
    description_en: str
    description_vi: str
    visual_cues_en: str
    visual_cues_vi: str
    tags: list[str] = Field(default_factory=list)


class MatchCandidate(BaseModel):
    node_id: str
    score: float


class MatchResult(BaseModel):
    node_id: str | None
    confidence: float
    top_k: list[MatchCandidate]


class Region(BaseModel):
    box: tuple[float, float, float, float]  # xyxy in pixel coords
    mask_png_b64: str                        # PNG-encoded binary mask
    query: str                               # detection query that produced this region
    score: float                             # detector score


class RegionDescription(BaseModel):
    query: str
    text: str


class CaptionResult(BaseModel):
    caption: str | None
    refusal: str | None
    debug: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests, verify they pass**

```bash
pytest tests/test_schemas.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add Pydantic models for inter-module values"
```

---

### Task 1.2: KB loader

**Files:**
- Create: `hanoi_caption/kb_loader.py`
- Test: `tests/test_kb_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kb_loader.py
from pathlib import Path

from hanoi_caption.kb_loader import load_kb


def test_load_kb_returns_dict_keyed_by_id(tmp_path: Path):
    kb_file = tmp_path / "kb.json"
    kb_file.write_text(
        '[{"id":"a","name_en":"A","name_vi":"a","type":"object",'
        '"parent_id":"categoryHaNoi","description_en":"d","description_vi":"d",'
        '"visual_cues_en":"v","visual_cues_vi":"v","tags":[]},'
        '{"id":"cat","name_en":"Cat","name_vi":"cat","type":"category",'
        '"parent_id":null,"description_en":"d","description_vi":"d",'
        '"visual_cues_en":"","visual_cues_vi":"","tags":[]}]'
    )
    nodes = load_kb(kb_file, only_objects=True)
    assert set(nodes.keys()) == {"a"}
    assert nodes["a"].name_en == "A"


def test_load_kb_real_sample():
    nodes = load_kb(Path("data/kb.json"), only_objects=True)
    assert len(nodes) == 15
    assert "temple_of_literature" in nodes
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_kb_loader.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `hanoi_caption/kb_loader.py`**

```python
"""Load the KB JSON file into a dict of KBNode."""
from __future__ import annotations

import json
from pathlib import Path

from hanoi_caption.schemas import KBNode


def load_kb(path: Path | str, only_objects: bool = True) -> dict[str, KBNode]:
    raw = json.loads(Path(path).read_text())
    nodes = [KBNode.model_validate(item) for item in raw]
    if only_objects:
        nodes = [n for n in nodes if n.type == "object"]
    return {n.id: n for n in nodes}
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_kb_loader.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/kb_loader.py tests/test_kb_loader.py
git commit -m "feat(kb_loader): load and filter KB JSON into KBNode dict"
```

---

### Task 1.3: Model registry skeleton

**Files:**
- Create: `hanoi_caption/model_registry.py`

This task creates the registry interface only — no models registered yet. Subsequent tasks register their models.

- [ ] **Step 1: Implement `hanoi_caption/model_registry.py`**

```python
"""Centralized lazy model loading + sequential eviction.

All other modules go through this registry to obtain models. The registry
keeps a working set under MAX_VRAM_GB by evicting LRU models when adding
a new one would exceed the budget. Eviction is voluntary: callers are
encouraged to call .evict(name) when they know they are done with a model
for a while.
"""
from __future__ import annotations

import gc
from collections import OrderedDict
from typing import Any, Callable

import torch

MAX_LOADED_MODELS = 6  # soft cap; primary control is per-stage evict() calls


class ModelRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, Callable[[], Any]] = {}
        self._loaded: OrderedDict[str, Any] = OrderedDict()

    def register(self, name: str, loader: Callable[[], Any]) -> None:
        if name in self._loaders:
            raise ValueError(f"Model '{name}' already registered")
        self._loaders[name] = loader

    def get(self, name: str) -> Any:
        if name not in self._loaders:
            raise KeyError(f"Model '{name}' not registered")
        if name in self._loaded:
            self._loaded.move_to_end(name)
            return self._loaded[name]
        if len(self._loaded) >= MAX_LOADED_MODELS:
            self._evict_oldest()
        model = self._loaders[name]()
        self._loaded[name] = model
        return model

    def evict(self, name: str) -> None:
        if name in self._loaded:
            del self._loaded[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def evict_all(self) -> None:
        names = list(self._loaded.keys())
        for n in names:
            self.evict(n)

    def loaded(self) -> list[str]:
        return list(self._loaded.keys())

    def _evict_oldest(self) -> None:
        oldest = next(iter(self._loaded))
        self.evict(oldest)


# Module-level singleton — every module imports this.
registry = ModelRegistry()
```

- [ ] **Step 2: Quick smoke test in a python REPL**

```bash
python -c "
from hanoi_caption.model_registry import registry
registry.register('toy', lambda: {'mock': 'model'})
print(registry.get('toy'))
print(registry.loaded())
registry.evict('toy')
print(registry.loaded())
"
```

Expected:
```
{'mock': 'model'}
['toy']
[]
```

- [ ] **Step 3: Commit**

```bash
git add hanoi_caption/model_registry.py
git commit -m "feat(model_registry): lazy-load + LRU evict singleton"
```

---

### Task 1.4: KB indexer (BGE-M3)

**Files:**
- Create: `hanoi_caption/kb_indexer.py`
- Test: `tests/test_kb_indexer.py`

- [ ] **Step 1: Write a unit test that exercises the cosine math without loading a real model**

```python
# tests/test_kb_indexer.py
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
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_kb_indexer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `hanoi_caption/kb_indexer.py`**

```python
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
```

- [ ] **Step 4: Run unit test, verify pass**

```bash
pytest tests/test_kb_indexer.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Manual smoke test (loads BGE-M3, ~1 GB download on first run)**

```bash
python -c "
from hanoi_caption.kb_loader import load_kb
from hanoi_caption.kb_indexer import build_or_load_index
nodes = load_kb('data/kb.json')
idx = build_or_load_index(nodes)
print('indexed', len(idx.node_ids), 'nodes, dim=', idx.embeddings.shape[1])
"
```

Expected: `indexed 15 nodes, dim= 1024`. A `data/cache/kb_index_<hash>.npz` is created.

- [ ] **Step 6: Commit**

```bash
git add hanoi_caption/kb_indexer.py tests/test_kb_indexer.py
git commit -m "feat(kb_indexer): BGE-M3 embeddings + cosine topk + .npz cache"
```

---

### Task 1.5: Image describer (Qwen2.5-VL-7B)

**Files:**
- Create: `hanoi_caption/image_describer.py`

Smoke-tested only (model-touching code; no pytest).

- [ ] **Step 1: Implement `hanoi_caption/image_describer.py`**

```python
"""Whole-image describer: produce a holistic visual description without naming landmarks."""
from __future__ import annotations

from PIL import Image

from hanoi_caption.model_registry import registry

MODEL_NAME = "qwen25_vl_7b"
HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "Describe what is visually present in this image. "
    "Mention architecture, materials, layout, surroundings, and people. "
    "Use 3 to 5 sentences. "
    "Do NOT name specific landmarks, places, cities, or countries."
)


def _load():
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(HF_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        load_in_4bit=True,
    )
    model.eval()
    return {"processor": processor, "model": model}


registry.register(MODEL_NAME, _load)


def describe_image(image: Image.Image) -> str:
    import torch

    bundle = registry.get(MODEL_NAME)
    processor, model = bundle["processor"], bundle["model"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    decoded = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    return decoded.strip()
```

- [ ] **Step 2: Manual smoke test (downloads ~6 GB on first run)**

Place a Hanoi landmark photo at `tests/fixtures/temple_of_literature_1.jpg` (any photo of the actual Temple of Literature; the exact framing does not matter for this smoke).

```bash
python -c "
from PIL import Image
from hanoi_caption.image_describer import describe_image
img = Image.open('tests/fixtures/temple_of_literature_1.jpg').convert('RGB')
print(describe_image(img))
"
```

Expected: a 3–5 sentence English paragraph describing the visible scene **without** the words "Temple of Literature" or "Hanoi". If the model leaks the landmark name, the prompt needs to be hardened — note as a follow-up but do not block the task.

- [ ] **Step 3: Commit**

```bash
git add hanoi_caption/image_describer.py
git commit -m "feat(image_describer): Qwen2.5-VL-7B holistic description"
```

---

### Task 1.6: KB matcher (cosine + VLM re-rank)

**Files:**
- Create: `hanoi_caption/kb_matcher.py`
- Test: `tests/test_kb_matcher.py`

The cosine + threshold logic is unit-testable with a fake re-ranker; the real VLM re-rank is exercised in the notebook.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kb_matcher.py
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
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_kb_matcher.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `hanoi_caption/kb_matcher.py`**

```python
"""Two-stage KB matching: cosine retrieval + VLM re-rank."""
from __future__ import annotations

import json
from typing import Callable

import numpy as np
from PIL import Image

from hanoi_caption.image_describer import registry as _img_registry  # noqa: F401  ensures VLM is registered
from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
from hanoi_caption.kb_indexer import KBIndex, embed_text
from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import KBNode, MatchCandidate, MatchResult

DEFAULT_THRESHOLD = 0.45
TOPK = 3


def _default_embed(text: str) -> np.ndarray:
    return embed_text([text])[0]


def _vlm_rerank(
    image: Image.Image,
    candidates: list[MatchCandidate],
    kb_nodes: dict[str, KBNode],
) -> tuple[str | None, float]:
    """Ask the VLM to choose among the top-k or say 'none'.

    Returns (node_id_or_None, confidence_in_[0,1]).
    """
    import torch

    bundle = registry.get(VLM_NAME)
    processor, model = bundle["processor"], bundle["model"]

    options_block = "\n".join(
        f"- id: {c.node_id} | name: {kb_nodes[c.node_id].name_en} | "
        f"visual cues: {kb_nodes[c.node_id].visual_cues_en[:300]}"
        for c in candidates
    )
    prompt = (
        "You are a Hanoi tour expert. Choose which landmark this image shows. "
        "Reply with strict JSON: {\"node_id\": <id-or-null>, \"confidence\": <0-1>}.\n\n"
        f"Options:\n{options_block}\n\n"
        "If none of the options match the image, return node_id=null."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    raw = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        node_id = parsed.get("node_id")
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        return (None, 0.0)
    if node_id not in {c.node_id for c in candidates}:
        return (None, confidence)
    return (node_id, confidence)


def match_kb(
    image: Image.Image,
    holistic_desc: str,
    kb_index: KBIndex,
    kb_nodes: dict[str, KBNode],
    threshold: float = DEFAULT_THRESHOLD,
    embed_fn: Callable[[str], np.ndarray] | None = None,
    rerank_fn: Callable[
        [Image.Image, list[MatchCandidate], dict[str, KBNode]],
        tuple[str | None, float],
    ]
    | None = None,
) -> MatchResult:
    embed_fn = embed_fn or _default_embed
    rerank_fn = rerank_fn or _vlm_rerank

    q = embed_fn(holistic_desc)
    candidates = kb_index.topk(q, k=TOPK)

    if not candidates or candidates[0].score < threshold:
        return MatchResult(node_id=None, confidence=candidates[0].score if candidates else 0.0, top_k=candidates)

    # The injected fake_rerank in tests has signature (image, candidate_ids, kb).
    # Real _vlm_rerank takes the same arity but with full candidates; pass full list.
    chosen, conf = rerank_fn(image, candidates, kb_nodes)
    return MatchResult(node_id=chosen, confidence=conf, top_k=candidates)
```

- [ ] **Step 4: Run unit tests, verify pass**

```bash
pytest tests/test_kb_matcher.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/kb_matcher.py tests/test_kb_matcher.py
git commit -m "feat(kb_matcher): cosine retrieval + VLM rerank with refusal gate"
```

---

### Task 1.7: Composer (Qwen2.5-7B-Instruct)

**Files:**
- Create: `hanoi_caption/composer.py`

Model-touching; smoke-tested in notebook. The prompt assembly is the testable part.

- [ ] **Step 1: Implement `hanoi_caption/composer.py`**

```python
"""Compose the final 150-300 word tour-guide caption."""
from __future__ import annotations

from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import KBNode, RegionDescription

MODEL_NAME = "qwen25_7b_instruct"
HF_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You are a tour guide writing for travelers. Voice: warm, observant, concrete. "
    "Write ONE paragraph of 150 to 300 words in English. "
    "Weave together (a) the historical and cultural facts provided and "
    "(b) the specific visual details actually observed in this photo. "
    "Do not invent facts beyond what is provided. Do not list — write prose. "
    "Do not mention that you are using a knowledge base or AI."
)


def _load():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_ID)
    model = AutoModelForCausalLM.from_pretrained(
        HF_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        load_in_4bit=True,
    )
    model.eval()
    return {"tokenizer": tokenizer, "model": model}


registry.register(MODEL_NAME, _load)


def build_user_prompt(
    kb_node: KBNode,
    region_descriptions: list[RegionDescription],
    holistic_desc: str,
) -> str:
    if region_descriptions:
        regions_block = "\n".join(
            f"- ({rd.query}) {rd.text}" for rd in region_descriptions
        )
    else:
        regions_block = "- (none — phase 1, no region grounding yet)"
    return (
        f"Landmark: {kb_node.name_en}\n\n"
        f"Background:\n{kb_node.description_en}\n\n"
        f"What is visible in this photo:\n{regions_block}\n\n"
        f"Holistic view: {holistic_desc}\n\n"
        "Write the paragraph now."
    )


def compose(
    kb_node: KBNode,
    region_descriptions: list[RegionDescription],
    holistic_desc: str,
) -> str:
    import torch

    bundle = registry.get(MODEL_NAME)
    tok, model = bundle["tokenizer"], bundle["model"]

    user_prompt = build_user_prompt(kb_node, region_descriptions, holistic_desc)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tok.decode(
        out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
```

- [ ] **Step 2: Add a unit test for prompt assembly**

```python
# tests/test_composer.py
from hanoi_caption.composer import build_user_prompt
from hanoi_caption.schemas import KBNode, RegionDescription


def _node():
    return KBNode(
        id="x", name_en="X Temple", name_vi="X", type="object",
        parent_id=None, description_en="X is old.", description_vi="",
        visual_cues_en="stone gate", visual_cues_vi="", tags=[],
    )


def test_prompt_includes_landmark_and_holistic():
    p = build_user_prompt(_node(), [], "A stone building with a courtyard.")
    assert "X Temple" in p
    assert "X is old." in p
    assert "A stone building" in p


def test_prompt_handles_empty_regions():
    p = build_user_prompt(_node(), [], "holistic")
    assert "phase 1" in p


def test_prompt_lists_regions():
    rds = [RegionDescription(query="gate", text="A red gate.")]
    p = build_user_prompt(_node(), rds, "holistic")
    assert "(gate) A red gate." in p
```

- [ ] **Step 3: Run tests, verify pass**

```bash
pytest tests/test_composer.py -v
```

Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add hanoi_caption/composer.py tests/test_composer.py
git commit -m "feat(composer): Qwen2.5-7B prompt + generation for final caption"
```

---

### Task 1.8: Phase 1 pipeline orchestration

**Files:**
- Create: `hanoi_caption/pipeline.py`
- Test: `tests/test_pipeline.py`

The pipeline has two functions: `caption_phase1` (no DAM) and (added later in Task 2.4) `caption_phase2`.

- [ ] **Step 1: Write the failing test (Phase 1 path with mocked stages)**

```python
# tests/test_pipeline.py
from PIL import Image

from hanoi_caption.pipeline import caption_phase1
from hanoi_caption.schemas import KBNode, MatchCandidate, MatchResult


def _img():
    return Image.new("RGB", (8, 8), color=(0, 0, 0))


def _kb():
    return {
        "a": KBNode(
            id="a", name_en="A", name_vi="A", type="object", parent_id=None,
            description_en="A description.", description_vi="",
            visual_cues_en="cue", visual_cues_vi="", tags=[],
        )
    }


def test_phase1_returns_caption_when_match_succeeds():
    res = caption_phase1(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,  # ignored when describe/match are mocked
        describe_fn=lambda im: "a black square",
        match_fn=lambda im, desc, idx, kb: MatchResult(
            node_id="a", confidence=0.9,
            top_k=[MatchCandidate(node_id="a", score=0.9)],
        ),
        compose_fn=lambda node, regions, desc: "A long paragraph about A.",
    )
    assert res.caption == "A long paragraph about A."
    assert res.refusal is None
    assert res.debug["match"]["node_id"] == "a"


def test_phase1_refuses_when_no_match():
    res = caption_phase1(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,
        describe_fn=lambda im: "?",
        match_fn=lambda im, desc, idx, kb: MatchResult(
            node_id=None, confidence=0.1, top_k=[],
        ),
        compose_fn=lambda *a, **k: "should not be called",
    )
    assert res.caption is None
    assert "Not a recognized" in res.refusal
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_pipeline.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement Phase 1 in `hanoi_caption/pipeline.py`**

```python
"""End-to-end orchestration."""
from __future__ import annotations

from typing import Callable

from PIL import Image

from hanoi_caption.kb_indexer import KBIndex
from hanoi_caption.schemas import (
    CaptionResult,
    KBNode,
    MatchResult,
    RegionDescription,
)

REFUSAL_TEXT = "Not a recognized Hanoi landmark."


def caption_phase1(
    image: Image.Image,
    kb_nodes: dict[str, KBNode],
    kb_index: KBIndex | None,
    describe_fn: Callable[[Image.Image], str] | None = None,
    match_fn: Callable[
        [Image.Image, str, KBIndex | None, dict[str, KBNode]], MatchResult
    ]
    | None = None,
    compose_fn: Callable[
        [KBNode, list[RegionDescription], str], str
    ]
    | None = None,
) -> CaptionResult:
    """KB-only path: image → describe → match → compose. No DAM/SAM/GroundingDINO."""
    if describe_fn is None:
        from hanoi_caption.image_describer import describe_image as describe_fn  # noqa
    if match_fn is None:
        from hanoi_caption.kb_matcher import match_kb as _match
        match_fn = lambda im, desc, idx, kb: _match(im, desc, idx, kb)
    if compose_fn is None:
        from hanoi_caption.composer import compose as compose_fn  # noqa

    debug: dict = {}

    holistic = describe_fn(image)
    debug["holistic_desc"] = holistic

    match = match_fn(image, holistic, kb_index, kb_nodes)
    debug["match"] = match.model_dump()

    if match.node_id is None:
        return CaptionResult(caption=None, refusal=REFUSAL_TEXT, debug=debug)

    node = kb_nodes[match.node_id]
    caption = compose_fn(node, [], holistic)
    debug["caption_chars"] = len(caption)
    return CaptionResult(caption=caption, refusal=None, debug=debug)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): Phase 1 orchestration (KB-only) with refusal gate"
```

---

### Task 1.9: Phase 1 notebook

**Files:**
- Create: `notebooks/01_phase1_kb_only.ipynb`

This is the v1 evaluation surface. Cells inspect every intermediate.

- [ ] **Step 1: Create the notebook with the cells below**

Use `jupyter lab` to create the file. Put one Python statement / block per cell.

**Cell 1 — imports:**
```python
import sys
sys.path.insert(0, "..")  # if running from notebooks/

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from hanoi_caption.kb_loader import load_kb
from hanoi_caption.kb_indexer import build_or_load_index
from hanoi_caption.image_describer import describe_image
from hanoi_caption.kb_matcher import match_kb
from hanoi_caption.composer import compose
from hanoi_caption.pipeline import caption_phase1
from hanoi_caption.model_registry import registry
```

**Cell 2 — load KB and index:**
```python
nodes = load_kb("../data/kb.json")
print(f"Loaded {len(nodes)} KB nodes:")
for nid, n in nodes.items():
    print(f"  {nid:35s} {n.name_en}")
kb_index = build_or_load_index(nodes)
print(f"\nKB index ready, {kb_index.embeddings.shape}")
```

**Cell 3 — pick a test image:**
```python
test_image = Path("../tests/fixtures/temple_of_literature_1.jpg")
img = Image.open(test_image).convert("RGB")
plt.figure(figsize=(8, 6))
plt.imshow(img)
plt.axis("off")
plt.show()
```

**Cell 4 — run Phase 1 pipeline:**
```python
result = caption_phase1(image=img, kb_nodes=nodes, kb_index=kb_index)
print("=== DEBUG ===")
import json
print(json.dumps(result.debug, indent=2, default=str)[:2000])
print("\n=== OUTPUT ===")
if result.caption:
    print(result.caption)
else:
    print("REFUSED:", result.refusal)
```

**Cell 5 — refusal smoke test:**
```python
# Use any non-Hanoi photo (e.g., your local park, kitchen, etc.)
neg_image = Path("../tests/fixtures/non_hanoi.jpg")
if neg_image.exists():
    nimg = Image.open(neg_image).convert("RGB")
    nres = caption_phase1(image=nimg, kb_nodes=nodes, kb_index=kb_index)
    print("expected refusal:", nres.refusal or "got caption (unexpected)")
else:
    print("Skipped — drop a non-Hanoi photo at", neg_image)
```

**Cell 6 — VRAM monitor:**
```python
import torch
print("loaded models:", registry.loaded())
print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM reserved : {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

- [ ] **Step 2: Run all cells. Verify**:
  - Cell 2 loads 15 nodes and prints index shape `(15, 1024)`.
  - Cell 4 prints a 150–300 word caption that mentions the matched landmark.
  - Cell 5 prints "expected refusal: Not a recognized Hanoi landmark." (assuming the negative image was provided).
  - Cell 6 reports VRAM ≤ 14 GB.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_phase1_kb_only.ipynb
git commit -m "feat(notebook): Phase 1 KB-only smoke test notebook"
```

---

## Phase 2 — Add DAM region grounding

### Task 2.1: Query extractor (Qwen2.5-7B)

**Files:**
- Create: `hanoi_caption/query_extractor.py`
- Test: `tests/test_query_extractor.py`

Reuses the Qwen2.5-7B already registered by the composer.

- [ ] **Step 1: Write the failing test (parsing logic only)**

```python
# tests/test_query_extractor.py
from hanoi_caption.query_extractor import parse_queries


def test_parse_queries_strips_markdown_fences():
    raw = '```json\n["a", "b", "c"]\n```'
    assert parse_queries(raw) == ["a", "b", "c"]


def test_parse_queries_handles_extra_prose():
    raw = 'Here are the queries: ["red gate", "stone stele"]. Done.'
    assert parse_queries(raw) == ["red gate", "stone stele"]


def test_parse_queries_dedupes_and_strips():
    raw = '["  Stele ", "stele", "Stele"]'
    out = parse_queries(raw)
    assert out == ["stele"]


def test_parse_queries_returns_empty_on_garbage():
    assert parse_queries("not json at all") == []
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_query_extractor.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `hanoi_caption/query_extractor.py`**

```python
"""Convert KB Visual Cues into short detector-friendly noun phrases."""
from __future__ import annotations

import json
import re

from hanoi_caption.composer import MODEL_NAME as LLM_NAME
from hanoi_caption.composer import _load as _llm_load  # noqa: F401  ensures registration
from hanoi_caption.model_registry import registry

EXTRACT_PROMPT = (
    "Extract 4 to 8 short noun phrases (1 to 4 words each) from the description below. "
    "Each phrase MUST name a physically detectable object that an open-vocabulary "
    "object detector can find in a photo (e.g., 'red gate', 'stone stele', 'tiered roof'). "
    "No verbs. No abstractions. No place names.\n\n"
    "Reply with ONLY a JSON array of strings. No prose, no markdown.\n\n"
    "Description:\n{desc}"
)


def parse_queries(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    m = re.search(r"\[.*?\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    seen = set()
    out: list[str] = []
    for it in items:
        if not isinstance(it, str):
            continue
        norm = it.strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def extract_queries(visual_cues_text: str) -> list[str]:
    import torch

    bundle = registry.get(LLM_NAME)
    tok, model = bundle["tokenizer"], bundle["model"]
    messages = [
        {"role": "user", "content": EXTRACT_PROMPT.format(desc=visual_cues_text)}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    raw = tok.decode(
        out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return parse_queries(raw)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_query_extractor.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/query_extractor.py tests/test_query_extractor.py
git commit -m "feat(query_extractor): KB cues -> JSON noun-phrase queries"
```

---

### Task 2.2: Region proposer (Grounding DINO + SAM 2)

**Files:**
- Create: `hanoi_caption/region_proposer.py`
- Test: `tests/test_region_proposer.py`

Pure-logic tests cover the IoU filter and fallback logic. The detection + segmentation models are smoke-tested in the notebook.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_region_proposer.py
import numpy as np

from hanoi_caption.region_proposer import filter_regions


class _Detection:
    def __init__(self, box, score, query):
        self.box = box
        self.score = score
        self.query = query


def _det(x1, y1, x2, y2, score=0.9, query="x"):
    return _Detection(box=(x1, y1, x2, y2), score=score, query=query)


def test_filter_drops_tiny_masks():
    image_area = 100 * 100
    dets = [_det(0, 0, 5, 5)]  # 25 px = 0.25% of image
    out = filter_regions(dets, image_area=image_area, min_area_frac=0.01, iou_threshold=0.7, max_keep=6)
    assert out == []


def test_filter_drops_high_iou_overlaps_keeping_higher_score():
    image_area = 100 * 100
    dets = [
        _det(10, 10, 50, 50, score=0.95),
        _det(11, 11, 51, 51, score=0.85),  # nearly identical -> dropped
        _det(60, 60, 90, 90, score=0.8),
    ]
    out = filter_regions(dets, image_area=image_area, min_area_frac=0.01, iou_threshold=0.7, max_keep=6)
    assert len(out) == 2
    assert any(d.score == 0.95 for d in out)
    assert all(d.score != 0.85 for d in out)


def test_filter_caps_max_keep():
    image_area = 100 * 100
    dets = [_det(0, 0, 30, 30 + i, score=0.9 - 0.01 * i) for i in range(10)]
    out = filter_regions(dets, image_area=image_area, min_area_frac=0.001, iou_threshold=0.99, max_keep=4)
    assert len(out) == 4
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_region_proposer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `hanoi_caption/region_proposer.py`**

```python
"""Detect KB-driven regions and segment them.

Grounding DINO (HuggingFace transformers integration) provides text-prompted
detection. SAM 2 converts boxes to binary masks. We filter masks for size
and IoU before passing to DAM.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from hanoi_caption.model_registry import registry

GDINO_NAME = "grounding_dino"
GDINO_HF = "IDEA-Research/grounding-dino-base"
SAM2_NAME = "sam2"
SAM2_HF = "facebook/sam2-hiera-base-plus"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
MIN_AREA_FRAC = 0.01
IOU_THRESHOLD = 0.7
MAX_KEEP = 6


def _load_gdino():
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(GDINO_HF)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GDINO_HF, torch_dtype=torch.float16
    ).to("cuda")
    model.eval()
    return {"processor": processor, "model": model}


def _load_sam2():
    # Uses Meta's sam2 package; install from PyPI as `sam2`.
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(SAM2_HF)
    return predictor


registry.register(GDINO_NAME, _load_gdino)
registry.register(SAM2_NAME, _load_sam2)


@dataclass
class _Detection:
    box: tuple[float, float, float, float]
    score: float
    query: str


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


def _box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def filter_regions(
    detections: list,
    image_area: float,
    min_area_frac: float = MIN_AREA_FRAC,
    iou_threshold: float = IOU_THRESHOLD,
    max_keep: int = MAX_KEEP,
) -> list:
    survivors = [d for d in detections if _box_area(d.box) / image_area >= min_area_frac]
    survivors.sort(key=lambda d: d.score, reverse=True)
    kept: list = []
    for d in survivors:
        if any(_iou(d.box, k.box) > iou_threshold for k in kept):
            continue
        kept.append(d)
        if len(kept) >= max_keep:
            break
    return kept


def _detect(image: Image.Image, queries: list[str]) -> list[_Detection]:
    if not queries:
        return []
    bundle = registry.get(GDINO_NAME)
    processor, model = bundle["processor"], bundle["model"]

    text_prompt = ". ".join(q.lower().strip() for q in queries) + "."
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    out: list[_Detection] = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        out.append(_Detection(box=(x1, y1, x2, y2), score=float(score), query=str(label)))
    return out


def _segment(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]:
    if not boxes:
        return []
    predictor = registry.get(SAM2_NAME)
    image_np = np.array(image)
    predictor.set_image(image_np)
    masks: list[np.ndarray] = []
    for box in boxes:
        m, _, _ = predictor.predict(box=np.array(box, dtype=np.float32), multimask_output=False)
        masks.append(m[0].astype(np.uint8))
    return masks


def _sam_automask_topk(image: Image.Image, k: int = 4) -> list[tuple[np.ndarray, tuple[float, float, float, float]]]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    sam = build_sam2(SAM2_HF)
    gen = SAM2AutomaticMaskGenerator(sam)
    masks = gen.generate(np.array(image))
    masks.sort(key=lambda m: m["area"], reverse=True)
    out = []
    for m in masks[:k]:
        x, y, w, h = m["bbox"]
        out.append((m["segmentation"].astype(np.uint8), (x, y, x + w, y + h)))
    return out


def _mask_to_b64_png(mask: np.ndarray) -> str:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def propose_regions(image: Image.Image, queries: list[str]):
    """Returns list[Region] (Pydantic) — see hanoi_caption.schemas."""
    from hanoi_caption.schemas import Region

    image_area = image.size[0] * image.size[1]

    dets = _detect(image, queries)
    kept = filter_regions(dets, image_area=image_area)

    if kept:
        masks = _segment(image, [d.box for d in kept])
        out: list[Region] = []
        for d, m in zip(kept, masks):
            out.append(Region(
                box=d.box, mask_png_b64=_mask_to_b64_png(m),
                query=d.query, score=d.score,
            ))
        return out

    # Fallback: SAM auto-mask top-K
    masks_with_boxes = _sam_automask_topk(image, k=4)
    return [
        Region(
            box=box, mask_png_b64=_mask_to_b64_png(m),
            query="(automask)", score=0.0,
        )
        for m, box in masks_with_boxes
    ]
```

- [ ] **Step 4: Run unit tests, verify pass**

```bash
pytest tests/test_region_proposer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Manual smoke test**

```bash
python -c "
from PIL import Image
from hanoi_caption.region_proposer import propose_regions
img = Image.open('tests/fixtures/temple_of_literature_1.jpg').convert('RGB')
regions = propose_regions(img, ['stone stele', 'tiered roof', 'courtyard gate'])
print(f'{len(regions)} regions')
for r in regions[:6]:
    print(f'  {r.query!r} score={r.score:.2f} box={tuple(round(v,1) for v in r.box)}')
"
```

Expected: at least 1 region for a real Temple of Literature photo. Print the queries that fired.

- [ ] **Step 6: Commit**

```bash
git add hanoi_caption/region_proposer.py tests/test_region_proposer.py
git commit -m "feat(region_proposer): GroundingDINO + SAM2 with IoU filter and automask fallback"
```

---

### Task 2.3: Region describer (DAM-3B)

**Files:**
- Create: `hanoi_caption/region_describer.py`

Smoke-tested only.

- [ ] **Step 1: Implement `hanoi_caption/region_describer.py`**

```python
"""DAM-3B focal description of masked regions.

DAM is installed from NVIDIA's repo:
    pip install git+https://github.com/NVlabs/describe-anything.git

The exact import path / API may differ between releases. The reference
quickstart at the time of writing exposes a high-level `DAMModel.describe`
that takes (PIL.Image, mask: np.ndarray, prompt: str | None) and returns
a string. If the API has changed by your install date, adapt _load and
describe_region to the current API. Treat that adaptation as part of
this task — do not invent helpers.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from hanoi_caption.model_registry import registry
from hanoi_caption.schemas import Region, RegionDescription

MODEL_NAME = "dam_3b"
DAM_PROMPT = (
    "Describe the highlighted region in 1 to 2 short sentences. "
    "Focus on visible attributes (material, color, condition, posture, action). "
    "Do not name the location or invent context outside the region."
)


def _load():
    # Adapt this import to the installed DAM release.
    from dam import DAMModel  # type: ignore

    model = DAMModel.from_pretrained("nvidia/DAM-3B")
    model.eval()
    model.to("cuda")
    return model


registry.register(MODEL_NAME, _load)


def _b64_png_to_mask(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return (np.array(img) > 127).astype(np.uint8)


def describe_regions(image: Image.Image, regions: list[Region]) -> list[RegionDescription]:
    if not regions:
        return []
    model = registry.get(MODEL_NAME)
    out: list[RegionDescription] = []
    for r in regions:
        mask = _b64_png_to_mask(r.mask_png_b64)
        text = model.describe(image=image, mask=mask, prompt=DAM_PROMPT)
        out.append(RegionDescription(query=r.query, text=text.strip()))
    return out
```

- [ ] **Step 2: Manual smoke test (run AFTER `pip install git+https://github.com/NVlabs/describe-anything.git`)**

```bash
python -c "
from PIL import Image
from hanoi_caption.region_proposer import propose_regions
from hanoi_caption.region_describer import describe_regions
img = Image.open('tests/fixtures/temple_of_literature_1.jpg').convert('RGB')
regions = propose_regions(img, ['stone stele', 'tiered roof'])
descs = describe_regions(img, regions[:2])
for d in descs:
    print(f'  ({d.query}) {d.text}')
"
```

Expected: each region described in 1–2 sentences.

If DAM's installed API differs from the imports above (`dam.DAMModel.from_pretrained`, `.describe`), adapt this module's `_load` and `describe_regions` to match the current quickstart and rerun. Do not block on missing methods — the public API is the contract that matters.

- [ ] **Step 3: Commit**

```bash
git add hanoi_caption/region_describer.py
git commit -m "feat(region_describer): DAM-3B focal description per masked region"
```

---

### Task 2.4: Phase 2 pipeline orchestration

**Files:**
- Modify: `hanoi_caption/pipeline.py` — add `caption_phase2`
- Modify: `tests/test_pipeline.py` — add Phase 2 tests

- [ ] **Step 1: Append failing tests**

```python
# tests/test_pipeline.py — append to the existing file
from hanoi_caption.pipeline import caption_phase2
from hanoi_caption.schemas import Region, RegionDescription


def test_phase2_evicts_models_in_order_and_returns_caption():
    calls: list[str] = []

    def describe_fn(im):
        calls.append("describe"); return "holistic"

    def match_fn(im, desc, idx, kb):
        calls.append("match"); return MatchResult(
            node_id="a", confidence=0.9,
            top_k=[MatchCandidate(node_id="a", score=0.9)],
        )

    def extract_fn(text):
        calls.append("extract"); return ["q1", "q2"]

    def propose_fn(im, queries):
        calls.append("propose")
        return [Region(box=(0,0,10,10), mask_png_b64="", query="q1", score=0.9)]

    def regions_describe_fn(im, regions):
        calls.append("regdesc")
        return [RegionDescription(query="q1", text="a thing")]

    def compose_fn(node, regions, desc):
        calls.append("compose")
        return "A " + " ".join(r.text for r in regions) + " paragraph."

    res = caption_phase2(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,
        describe_fn=describe_fn,
        match_fn=match_fn,
        extract_queries_fn=extract_fn,
        propose_regions_fn=propose_fn,
        describe_regions_fn=regions_describe_fn,
        compose_fn=compose_fn,
    )
    assert res.caption is not None and "a thing" in res.caption
    assert calls == ["describe", "match", "extract", "propose", "regdesc", "compose"]


def test_phase2_refusal_path_skips_dam():
    calls: list[str] = []

    def fail(*a, **k):
        calls.append("should_not_be_called"); raise AssertionError

    res = caption_phase2(
        image=_img(),
        kb_nodes=_kb(),
        kb_index=None,
        describe_fn=lambda im: "?",
        match_fn=lambda im, d, idx, kb: MatchResult(node_id=None, confidence=0.0, top_k=[]),
        extract_queries_fn=fail,
        propose_regions_fn=fail,
        describe_regions_fn=fail,
        compose_fn=fail,
    )
    assert res.caption is None and "Not a recognized" in res.refusal
    assert calls == []
```

- [ ] **Step 2: Run test, confirm failure**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 2 new tests fail with ImportError on `caption_phase2`.

- [ ] **Step 3: Append `caption_phase2` to `hanoi_caption/pipeline.py`**

```python
# Append to hanoi_caption/pipeline.py

from hanoi_caption.schemas import Region


def caption_phase2(
    image: Image.Image,
    kb_nodes: dict[str, KBNode],
    kb_index: KBIndex | None,
    describe_fn: Callable[[Image.Image], str] | None = None,
    match_fn: Callable[
        [Image.Image, str, KBIndex | None, dict[str, KBNode]], MatchResult
    ]
    | None = None,
    extract_queries_fn: Callable[[str], list[str]] | None = None,
    propose_regions_fn: Callable[[Image.Image, list[str]], list[Region]] | None = None,
    describe_regions_fn: Callable[
        [Image.Image, list[Region]], list[RegionDescription]
    ]
    | None = None,
    compose_fn: Callable[
        [KBNode, list[RegionDescription], str], str
    ]
    | None = None,
) -> CaptionResult:
    """Full pipeline: identify -> queries -> detect/segment/describe -> compose.

    Manages model eviction between stages to stay under the 14 GB working-set budget.
    """
    if describe_fn is None:
        from hanoi_caption.image_describer import describe_image as describe_fn  # noqa
    if match_fn is None:
        from hanoi_caption.kb_matcher import match_kb as _match
        match_fn = lambda im, desc, idx, kb: _match(im, desc, idx, kb)
    if extract_queries_fn is None:
        from hanoi_caption.query_extractor import extract_queries as extract_queries_fn  # noqa
    if propose_regions_fn is None:
        from hanoi_caption.region_proposer import propose_regions as propose_regions_fn  # noqa
    if describe_regions_fn is None:
        from hanoi_caption.region_describer import describe_regions as describe_regions_fn  # noqa
    if compose_fn is None:
        from hanoi_caption.composer import compose as compose_fn  # noqa

    from hanoi_caption.model_registry import registry

    debug: dict = {}

    # Stage 3 — describe
    holistic = describe_fn(image)
    debug["holistic_desc"] = holistic

    # Stage 4 — match (uses VLM re-rank, then we are done with the VLM)
    match = match_fn(image, holistic, kb_index, kb_nodes)
    debug["match"] = match.model_dump()
    if match.node_id is None:
        return CaptionResult(caption=None, refusal=REFUSAL_TEXT, debug=debug)

    node = kb_nodes[match.node_id]

    # Stage 5 — extract queries (loads Qwen2.5-7B; keep it for stage 8)
    queries = extract_queries_fn(node.visual_cues_en)
    debug["queries"] = queries

    # Free the VLM before loading detection stack
    try:
        from hanoi_caption.image_describer import MODEL_NAME as VLM_NAME
        registry.evict(VLM_NAME)
    except Exception:
        pass

    # Stage 6 — propose regions
    regions = propose_regions_fn(image, queries)
    debug["n_regions"] = len(regions)
    debug["regions"] = [r.model_dump(exclude={"mask_png_b64"}) for r in regions]

    # Stage 7 — describe regions
    region_descs = describe_regions_fn(image, regions)
    debug["region_descriptions"] = [rd.model_dump() for rd in region_descs]

    # Free detection stack before composing
    for n in ("grounding_dino", "sam2", "dam_3b", "bge_m3"):
        try:
            registry.evict(n)
        except Exception:
            pass

    # Stage 8 — compose
    caption = compose_fn(node, region_descs, holistic)
    debug["caption_chars"] = len(caption)
    return CaptionResult(caption=caption, refusal=None, debug=debug)
```

- [ ] **Step 4: Run tests, verify pass**

```bash
pytest tests/test_pipeline.py -v
```

Expected: 4 passed (2 Phase 1 + 2 Phase 2).

- [ ] **Step 5: Commit**

```bash
git add hanoi_caption/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): Phase 2 full pipeline with sequential model eviction"
```

---

### Task 2.5: Phase 2 notebook

**Files:**
- Create: `notebooks/02_phase2_full_pipeline.ipynb`

- [ ] **Step 1: Create the notebook with the cells below**

**Cell 1 — imports + KB:**
```python
import sys; sys.path.insert(0, "..")
from pathlib import Path
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt

from hanoi_caption.kb_loader import load_kb
from hanoi_caption.kb_indexer import build_or_load_index
from hanoi_caption.pipeline import caption_phase1, caption_phase2
from hanoi_caption.model_registry import registry

nodes = load_kb("../data/kb.json")
kb_index = build_or_load_index(nodes)
print(f"KB ready: {len(nodes)} nodes")
```

**Cell 2 — pick 3 photos of the same landmark:**
```python
photos = [
    Path("../tests/fixtures/temple_of_literature_1.jpg"),
    Path("../tests/fixtures/temple_of_literature_2.jpg"),
    Path("../tests/fixtures/temple_of_literature_3.jpg"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, p in zip(axes, photos):
    ax.imshow(Image.open(p)); ax.set_title(p.name); ax.axis("off")
plt.show()
```

**Cell 3 — Phase 1 captions for all 3:**
```python
phase1_results = []
for p in photos:
    img = Image.open(p).convert("RGB")
    r = caption_phase1(image=img, kb_nodes=nodes, kb_index=kb_index)
    phase1_results.append(r)
    print(f"\n--- Phase 1: {p.name} ---")
    print(r.caption or r.refusal)
```

**Cell 4 — Phase 2 captions for all 3:**
```python
phase2_results = []
for p in photos:
    img = Image.open(p).convert("RGB")
    r = caption_phase2(image=img, kb_nodes=nodes, kb_index=kb_index)
    phase2_results.append(r)
    print(f"\n--- Phase 2: {p.name} ---")
    print(r.caption or r.refusal)
    print(f"  regions: {r.debug.get('n_regions')}, queries: {r.debug.get('queries')}")
```

**Cell 5 — sanity: Phase 2 outputs should be visibly different across photos:**
```python
def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0

print("Phase 1 pairwise word-jaccard:")
print(f"  1-2: {jaccard(phase1_results[0].caption, phase1_results[1].caption):.2f}")
print(f"  1-3: {jaccard(phase1_results[0].caption, phase1_results[2].caption):.2f}")
print(f"  2-3: {jaccard(phase1_results[1].caption, phase1_results[2].caption):.2f}")
print("\nPhase 2 pairwise word-jaccard:")
print(f"  1-2: {jaccard(phase2_results[0].caption, phase2_results[1].caption):.2f}")
print(f"  1-3: {jaccard(phase2_results[0].caption, phase2_results[2].caption):.2f}")
print(f"  2-3: {jaccard(phase2_results[1].caption, phase2_results[2].caption):.2f}")
```

Expected: Phase 2 jaccards are noticeably lower than Phase 1 jaccards. This is the qualitative win.

**Cell 6 — visualize masks for the first photo:**
```python
import base64, io, numpy as np
img = Image.open(photos[0]).convert("RGB")
result = caption_phase2(image=img, kb_nodes=nodes, kb_index=kb_index)
regions = result.debug["regions"]
print(f"queries fired: {result.debug['queries']}")
print(f"regions kept : {len(regions)}")
for r in regions[:6]:
    print(f"  {r['query']!r} score={r['score']:.2f}")
```

**Cell 7 — VRAM check:**
```python
print("loaded:", registry.loaded())
print(f"VRAM allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
```

- [ ] **Step 2: Run all cells. Verify**:
  - Phase 2 word-jaccards < Phase 1 word-jaccards (Phase 2 captions are visibly more photo-specific).
  - Peak VRAM < 14 GB.
  - At least one region fires per photo (or the automask fallback returned 4).

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_phase2_full_pipeline.ipynb
git commit -m "feat(notebook): Phase 2 full pipeline + Phase 1 vs Phase 2 comparison"
```

---

## Final Validation

### Task F.1: Full test sweep

- [ ] **Step 1: Run the unit-test suite**

```bash
pytest -v
```

Expected: every test passes. Total runtime < 10 s (no model-touching tests).

- [ ] **Step 2: Run both notebooks top-to-bottom**

In JupyterLab, restart kernel and run all cells in `01_phase1_kb_only.ipynb`, then in `02_phase2_full_pipeline.ipynb`. No exceptions, no OOM.

- [ ] **Step 3: Verify peak VRAM**

`02_phase2_full_pipeline.ipynb` Cell 7 must report peak VRAM ≤ 14 GB. If exceeded, revisit `caption_phase2` and evict `MODEL_NAME` (Qwen2.5-VL) immediately after `match_fn` returns instead of after `extract_queries_fn`.

### Task F.2: Spec-coverage self-review

- [ ] **Step 1: Walk the spec sections and confirm coverage**

| Spec § | Implemented in |
|---|---|
| §3.1 Pipeline data flow | Tasks 1.5, 1.6, 1.8, 2.1–2.4 |
| §3.3 Latency budget | Task F.1 manual timing in notebook (no test) |
| §4 Components & IO | Tasks 1.1–1.7, 2.1–2.3 |
| §5 KB schema & integration | Tasks 1.1, 1.2, 0.3 |
| §6 Decision gates / refusal | Task 1.6 (kb_matcher), 1.8 (pipeline), 2.4 (pipeline) |
| §7.1 Phase 1 | Tasks 1.1–1.9 |
| §7.2 Phase 2 | Tasks 2.1–2.5 |
| §8 Project layout | Task 0.1 |
| §9 Dependencies | Task 0.1 (`pyproject.toml`), README setup commands |
| §10.1 Memory plan | Task 2.4 (eviction calls), Task F.1 (VRAM check) |
| §10.2 Blackwell setup | Task 0.2 (CUDA-capability check) |
| §11 Risks | Documented in spec; no implementation work needed |
| §12 Out of scope | Confirmed not implemented |

If any spec requirement above is unimplemented, return to the relevant task before declaring complete.

---
