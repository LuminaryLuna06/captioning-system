# Retriever Backbone Comparison — Design

**Date:** 2026-05-25
**Status:** approved, ready for implementation plan
**Goal:** A plug-and-play ("lego") notebook for qualitative comparison of four image-embedding backbones in the per-frame retrieval step of the video captioning pipeline.

## Context

The current pipeline (`hanoi_caption/video_pipeline.py`) retrieves a landmark `kb_id` for each sampled video frame via a single backbone — DINOv3-vits16 — wrapped in `scripts/data_collection/feature_extractor.py:FeatureExtractor` and indexed in `scripts/data_collection/indexer.py`. We want to compare retrieval quality and behavior across four architectural paradigms before any commitment to a different backbone for production.

Segmentation evaluation is not yet complete, so end-to-end quantitative metrics are out of scope. The comparison this design supports is **qualitative**: side-by-side top-K matches and per-frame timeline visualizations across backbones, on fixed inputs.

## Scope

In scope:
- New module `hanoi_caption/retrieval/` containing a `BackboneExtractor` protocol, four concrete extractors, a cache-aware index builder, and retrieval closures.
- A CLI to pre-build the FAISS cache for all four backbones (one-time per backbone).
- A new notebook `notebooks/03_retriever_comparison.ipynb` that loads all four extractors + caches once and runs two visualization sections.
- Unit tests for index/retrieve logic (mock extractor) and slow tests for real-load smoke contracts of each backbone.

Out of scope:
- Quantitative benchmarking against ground-truth segmentation (deferred until `scripts/eval/eval_segmentation.py` is wired up against this).
- Changing the production pipeline's default retriever — `_default_retrieve_fn` and `FeatureExtractor`/`ImageIndexer` stay untouched.
- Re-ranking, multi-stage retrieval, ensemble — single nearest-neighbor only.
- Notebook tests (nbval).

## Models compared

| Slot | HF checkpoint | Paradigm | Embedding source | Expected dim |
|---|---|---|---|---|
| DINOv3 | `facebook/dinov3-vits16-pretrain-lvd1689m` | Self-supervised (image-only) | `last_hidden_state[:, 0]` (CLS) | 384 |
| ResNet-50 | `microsoft/resnet-50` | Supervised ImageNet, convolutional | `pooler_output` (global avg pool, flattened) | 2048 |
| SigLIP-2 | `google/siglip2-base-patch16-224` | Sigmoid image-text contrastive (vision tower only) | `pooler_output` | 768 |
| ViT | `google/vit-base-patch16-224` | Supervised ImageNet, transformer | `last_hidden_state[:, 0]` (CLS) | 768 |

Rationale: cover four distinct pretraining regimes (self-supervised vs supervised vs contrastive; conv vs transformer) so qualitative differences are interpretable as paradigm differences, not noise.

## Architecture

### Module layout

```
hanoi_caption/
  retrieval/                        ← NEW
    __init__.py
    backbones.py                    ← BackboneExtractor protocol + _HFExtractor base + 4 concrete classes
    index.py                        ← build_or_load_index(extractor, kb_dir, cache_dir)
    retrieve.py                     ← make_retrieve_fn, make_topk_fn

scripts/data_collection/
  build_all_backbones.py            ← NEW CLI: build cache once per backbone
  extract_fixed_frames.py           ← NEW one-shot helper: decode fixed (video, t, kb_id) triples into tests/fixtures/

notebooks/
  03_retriever_comparison.ipynb     ← NEW

tests/
  unit/retrieval/
    test_index.py                   ← NEW
    test_retrieve.py                ← NEW
  models/
    test_backbones.py               ← NEW (slow, GPU)

data/cache/                         ← gitignored
  dinov3_vits16/{faiss.index, id_map.json}
  resnet50/{...}
  siglip2_base/{...}
  vit_base/{...}

tests/fixtures/retriever_frames/    ← gitignored
  <kb_id>.jpg                       ← one frame per representative landmark
```

The existing pipeline modules (`hanoi_caption/video_pipeline.py`, `scripts/data_collection/feature_extractor.py`, `scripts/data_collection/indexer.py`) are not modified. The new `retrieval/` module is fully independent and will be wireable into `caption_video(retrieve_fn=...)` later if a backbone change is approved.

### BackboneExtractor protocol

```python
class BackboneExtractor(Protocol):
    name: str       # short id, used as cache subdirectory: "dinov3_vits16", ...
    dim: int        # embedding dimension (auto-detected at __init__ via dummy forward)

    def extract(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return float32 array shape (len(images), dim), L2-normalized along dim=1."""
```

Single method, two attributes. Every backbone-specific concern (processor choice, which output to pool, model class) is internal to the concrete class.

### Concrete extractors

All four share a `_HFExtractor` base (in the same file) that handles:
- Loading `AutoImageProcessor` + the appropriate model class
- Moving to CUDA (this project targets a single 16 GB RTX — no CPU fallback path)
- `torch.no_grad()` forward + L2 normalize + `.cpu().numpy().astype("float32")`
- `dim` detection via one dummy forward in `__init__`

Each concrete class overrides only `_embed(outputs) -> torch.Tensor`:

| Class | `name` | `_embed(outputs)` |
|---|---|---|
| `Dinov3Extractor` | `"dinov3_vits16"` | `outputs.last_hidden_state[:, 0]` |
| `Resnet50Extractor` | `"resnet50"` | `outputs.pooler_output.flatten(1)` |
| `Siglip2Extractor` | `"siglip2_base"` | `outputs.pooler_output` |
| `VitExtractor` | `"vit_base"` | `outputs.last_hidden_state[:, 0]` |

SigLIP-2 loads through its vision-tower-only class (e.g. `Siglip2VisionModel` if available in the installed `transformers` version, otherwise the vision module of `AutoModel`) so the text tower is not allocated.

### Index cache

`build_or_load_index(extractor, kb_images_dir, cache_dir, batch_size=16, force_rebuild=False) -> (faiss.Index, dict[int, str])`:

1. `cache_path = cache_dir / extractor.name`.
2. If `cache_path/faiss.index` and `cache_path/id_map.json` exist and `not force_rebuild`: `faiss.read_index` + load JSON → return.
3. Else:
   - Walk `kb_images_dir` for `.jpg/.jpeg/.png/.bmp/.webp`.
   - Batch-extract via `extractor.extract`.
   - Build `faiss.IndexFlatIP(extractor.dim)` (cosine ≡ inner product on L2-normalized vectors).
   - `id_map = {row_id: image_path}`; `kb_id = basename(dirname(path))`.
   - Write index + JSON; log progress every batch.

Decision: **Flat IP only**, no HNSW / scalar quantization. KB is small (a few hundred to a few thousand images); a flat exact baseline keeps comparison clean (no approximation artifacts). Different from the existing `ImageIndexer` which exposes HNSW/SQ — this is experimental code with different priorities.

### Retrieve closures

```python
def make_retrieve_fn(extractor, index, id_map) -> Callable[[Image.Image], tuple[str | None, float]]:
    """Returns a closure for k=1 retrieval. Drop-in for caption_video(retrieve_fn=...)."""

def make_topk_fn(extractor, index, id_map, k: int) -> Callable[[Image.Image], list[dict]]:
    """Returns a closure that yields [{path, kb_id, score}, ...] of length k.
    For top-K visualization in the notebook."""
```

`make_retrieve_fn` mirrors the contract of `_default_retrieve_fn` in `video_pipeline.py` so the new module is pipeline-compatible.

### CLI: `scripts/data_collection/build_all_backbones.py`

```
python scripts/data_collection/build_all_backbones.py \
    [--backbones dinov3,resnet50,siglip2,vit]   # default: all 4
    [--kb-dir data/kb_images]
    [--cache-dir data/cache]
    [--force]
```

Sequentially instantiates each requested backbone, calls `build_or_load_index`, prints summary. Does not eagerly unload between models — the four together fit comfortably in 16 GB VRAM (~1-2 GB peak).

### One-shot helper: `extract_fixed_frames.py`

A short script with a hard-coded list of `(video_path, timestamp_s, kb_id)` triples covering five representative landmarks (Nha Tho Lon, Nha Hat Lon, Nha Khach Chinh Phu, Den Ngoc Son, Bao Tang Gom). Decodes one frame per triple via `hanoi_caption.video_pipeline.sample_frames` (or direct OpenCV seek) and writes `tests/fixtures/retriever_frames/<kb_id>.jpg`. Run once; the notebook reads the resulting JPGs.

## Notebook structure

`notebooks/03_retriever_comparison.ipynb`, ~6 cells:

1. **Imports + config** — paths, constants (`TOPK=5`, `SAMPLE_FPS=1.0`, `TIMELINE_VIDEO=tests/videos/NhaThoLon_S_T03.MOV`).
2. **Load all four extractors + indexes once** —
   ```python
   EXTRACTORS = {"dinov3": Dinov3Extractor(), "resnet50": Resnet50Extractor(),
                 "siglip2": Siglip2Extractor(), "vit": VitExtractor()}
   INDEXES = {n: build_or_load_index(e, KB_DIR, CACHE_DIR) for n, e in EXTRACTORS.items()}
   ```
   Kept in memory for the rest of the session — no swap-out between sections.
3. **Visualization helpers** — `show_topk_grid(...)` and `show_timeline(...)` matplotlib functions inline (~50 lines). Notebook-local because they are purely presentational.
4. **Section A — Top-K view.** For each fixed frame in `tests/fixtures/retriever_frames/`, run `make_topk_fn(k=5)` on each backbone, render a grid (rows = queries, cols = per-model top-5 with thumbnail + kb_id + score).
5. **Section B — Timeline view.** Sample `TIMELINE_VIDEO` at 1 fps via `sample_frames`, run `make_retrieve_fn` on each backbone, render stacked horizontal bands (one per model, x = time, color = predicted kb_id, opacity = score).
6. **Quick stats (optional)** — markdown table: latency per frame per model, embedding dim, fraction of frames with score > 0.5. Not a real benchmark; just a sanity signal until ground-truth segmentation lands.

`TIMELINE_VIDEO` is a top-level variable so the user can point it at any other clip without editing the rest of the notebook.

## Data flow

```
                                    ┌─────────────────┐
data/kb_images/<kb>/*.jpg ─────────►│ build_or_load_  │
                                    │     index       │◄── extractor.extract (batched)
                                    └────────┬────────┘
                                             │
                                             ▼
                                  data/cache/<name>/
                                    faiss.index
                                    id_map.json
                                             │
              ┌──────────────────────────────┴──────────────────────────────┐
              │                                                             │
              ▼                                                             ▼
┌──────────────────────────────┐                          ┌──────────────────────────────┐
│  make_topk_fn(k=5)           │                          │  make_retrieve_fn            │
│  ↑                           │                          │  ↑                           │
│ tests/fixtures/retriever_    │                          │ sample_frames(TIMELINE_      │
│   frames/<kb>.jpg            │                          │   VIDEO, fps=1)              │
└──────────────┬───────────────┘                          └──────────────┬───────────────┘
               │                                                          │
               ▼                                                          ▼
        show_topk_grid                                              show_timeline
       (Section A output)                                          (Section B output)
```

## Error handling

- `build_or_load_index` raises if `kb_images_dir` has zero indexable files (clear message: misconfigured KB dir).
- Concrete extractors raise immediately if the HF download fails (no silent fallback to CPU-only or to a different model).
- `make_retrieve_fn` returns `(None, 0.0)` when FAISS returns index `-1` (mirrors existing `_default_retrieve_fn` behavior) — caller decides what "unknown" means.
- The notebook does no defensive error handling; failures should surface as Python tracebacks for diagnosis.

## Testing strategy

Test files mirror the new module layout:

### `tests/unit/retrieval/test_index.py` (fast, mock extractor)

- `FakeExtractor` with `name="fake"`, `dim=4`, `extract` returns deterministic float32 from filename hash. No model loads.
- `test_build_creates_cache` — first call writes `faiss.index` + `id_map.json`; `id_map` keys match FAISS row indices and values are the image paths.
- `test_load_skips_rebuild` — monkey-patch `extract` to raise; second call doesn't raise (cache hit).
- `test_force_rebuild_calls_extract` — `force_rebuild=True` triggers re-extraction.
- `test_search_returns_nearest` — query an image identical to one already indexed → `index.search(k=1)` returns its row id with score ≈ 1.

### `tests/unit/retrieval/test_retrieve.py` (fast, mock everything)

- `test_make_retrieve_fn_returns_kb_id` — fake index + `id_map = {0: "x/y/kb_abc/img.jpg"}` → `_retrieve(any_img)` returns `("kb_abc", float)`.
- `test_returns_none_when_idx_negative` — patch `index.search` → `(-1, 0.0)` → `(None, 0.0)`.
- `test_make_topk_fn_returns_k_results` — `k=3` closure returns list of length 3 with `{path, kb_id, score}` per element.

### `tests/models/test_backbones.py` (slow, GPU)

One parametrized test over `[(Dinov3Extractor, 384), (Resnet50Extractor, 2048), (Siglip2Extractor, 768), (VitExtractor, 768)]`:

- `ext.dim == expected_dim`
- `ext.extract([dummy, dummy]).shape == (2, expected_dim)` and dtype `float32`
- Row L2 norms ≈ 1.0 (tolerance 1e-5)
- Determinism: two calls with the same input return the same embedding (model is in `eval()` mode)

Adding a fifth backbone later is one line in `parametrize`.

### Not tested

- The notebook itself (it is glue + matplotlib; logic lives in the source module which is tested).
- `build_all_backbones.py` CLI (thin wrapper; smoke-tested by running it once).
- `extract_fixed_frames.py` (one-shot, not on the hot path).

## Test commands

```bash
pytest tests/unit/retrieval         # < 2s, every commit
pytest tests/models -m slow         # GPU, when changing a backbone
pytest tests/ -m "not slow"         # CI default
```

## Risks / open questions

- **Image preprocessing differences.** Each `AutoImageProcessor` resizes/normalizes per its model's training (DINOv3 uses 224, ResNet uses 224 with ImageNet stats, SigLIP-2 uses 224 with its own stats, ViT uses 224). This is the *intended* comparison: each backbone gets its proper preprocessing. We do not try to standardize input — that would handicap models trained with different statistics.
- **Embedding dimension mismatch.** Models have different `dim` (384 / 768 / 2048). This is fine because each builds its own FAISS index of matching dimension; we never mix embeddings across backbones.
- **Score scale comparability.** Cosine similarity is bounded in `[-1, 1]` for all four. Absolute scores are still not strictly comparable across backbones (different feature geometries), but rank order and within-backbone score distributions are. The Section B "score > 0.5" stat in cell 6 is a rough signal per backbone, not a cross-backbone metric.
- **Fixed-frames extraction depends on local videos.** `extract_fixed_frames.py` references files in `tests/videos/` (gitignored). On a fresh checkout, the user would need to run this once after pulling videos. Document this in the notebook's first markdown cell.
