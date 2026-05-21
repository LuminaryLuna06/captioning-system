# CLAUDE.md

Hanoi landmark captioning — video only. KB-grounded captions (150–300 words, tour-guide English) per segment of walking-tour clips.

## Environment

- Python: `D:\Jupiter\luna_env\Scripts\python.exe` (luna_env, PyTorch + CUDA 12.8 for Blackwell sm_120). Base Anaconda Python has a DLL conflict — do not use it.

## Pipeline — `hanoi_caption/video_pipeline.py`

1. **Sample** — decode frames at `sample_fps` (default 1.0) via OpenCV.
2. **Retrieve per frame** — DINOv3 features → FAISS cosine top-1 against a pre-built reference-image index → `(kb_id, score)`. Index + `id_map.json` live under `data/cache/`.
3. **Smooth & group** — sliding-window majority vote over kb_ids, merge consecutive same-id frames into runs, absorb short runs into the longer known neighbor, drop unknown segments shorter than `min_segment_seconds`.
4. **Caption per segment** — `pick_frame_indices` evenly samples K frames within `dam_frame_budget=(4,8)`; DAM-3B receives all K frames + all-ones masks and writes one KB-grounded paragraph naming the landmark.

Public entrypoint: `caption_video(video_path, kb_nodes, dino_index_path, id_map_path, ...)` → `list[VideoSegment]`.

## Modules

- `hanoi_caption/video_pipeline.py` — entrypoint + smoothing/grouping/DAM-prompt helpers.
- `hanoi_caption/dam_model.py` — DAM-3B loader, registered with `model_registry`.
- `hanoi_caption/kb_loader.py` — `load_kb(path)`, `index_by_kb_id(nodes)`.
- `hanoi_caption/model_registry.py` — lazy model loading with LRU eviction under a VRAM budget.
- `hanoi_caption/schemas.py` — `KBNode`, `VideoSegment`.

## Knowledge base

- `data/kb.json` — bilingual KB nodes (`kb_id`, `name_en`, `description_en`, `visual_cues_en`, ...).
- `data/kb_images/<kb_id>/*.jpg` — reference images per landmark (gitignored). Used to build the DINOv3 index via `scripts/data_collection/`.
- `data/cache/dino_faiss.index`, `data/cache/id_map.json` — built artifacts (gitignored).

## Notebook

- `notebooks/02_phase2_full_pipeline.ipynb` — interactive demo: setup → `caption_video` on a clip → timeline + caption render.

## Data collection — `scripts/data_collection/`

Pipeline for building the DINOv3 reference index that `caption_video` retrieves against:

- `config.py` — central config (model IDs, paths). Default retriever = `facebook/dinov3-vits16-pretrain-lvd1689m`; heavy reranker = `dinov3-vitb16`. Paths resolve to `data/kb_images/`, `data/cache/dino_faiss.index`, `data/cache/id_map.json`.
- `crawler.py` — `WikimediaCrawler`: fetches landmark images from Wikimedia Commons (Wikimedia-policy User-Agent).
- `crawl_selected.py` — CLI on top of `WikimediaCrawler`. Curated per-`kb_id` queries (e.g. `"Bat Trang pottery"` instead of bare `"Pottery"`). Run: `python scripts/data_collection/crawl_selected.py [--limit 12] [--group all|2|3|8|9|11]` → writes `data/kb_images/<kb_id>/*.jpg`.
- `feature_extractor.py` — `FeatureExtractor`: wraps DINOv3 (`AutoImageProcessor` + `AutoModel`), returns L2-normalized CLS embeddings. Also imported by `video_pipeline._default_retrieve_fn` at runtime.
- `indexer.py` — `ImageIndexer`: walks `data/kb_images/`, extracts features, builds FAISS index, writes `dino_faiss.index` + `id_map.json` (FAISS row → image path). Run: `python scripts/data_collection/indexer.py data/kb_images`.

**Order of operations** (only when adding/changing landmarks):

```bash
python scripts/data_collection/crawl_selected.py --limit 12   # 1. download images
python scripts/data_collection/indexer.py data/kb_images      # 2. build FAISS index
```

After this, `caption_video` can use the updated `data/cache/dino_faiss.index` + `id_map.json` without code changes.

## Evaluation — `scripts/eval/`

Run in order:

1. `generate_landmark_map.py` → `data/eval/landmark_map.json`
2. `build_test_set.py --video-dir tests/videos` → `data/eval/test_set.json`
3. `run_pipeline.py` → `data/eval/pipeline_results.json`
4. `eval_segmentation.py` → `data/eval/seg_metrics.json`
5. `eval_caption.py` → `data/eval/caption_metrics.json` (BLEU / ROUGE / …)
6. `eval_llm_judge.py` → `data/eval/llm_scores.json`
7. `eval_summary.py` → `data/eval/eval_summary.{csv,tex}`

Per-experiment snapshots go under `data/eval/runs/<date>_<label>/`.

## Tests — `tests/`

Organized by **test type**, not by source module, so adding model variants (Qwen-VL, CLIP, alternative DAM, …) only touches `tests/models/`:

```
tests/
  unit/                    fast, no GPU; pure logic
    test_kb_loader.py
    test_schemas.py
    video_pipeline/        per-function unit tests
      test_smoothing.py
      test_frame_budget.py
      test_dam_caption_prompt.py
  integration/             multi-module, mocked models
    test_caption_video_smoke.py
  models/                  per-model tests (real load, @slow) — empty for now
  eval/                    eval script tests
  fixtures/, videos/       gitignored
```

Selective runs:

```bash
pytest tests/unit                # every commit, < 3s
pytest tests/integration         # every commit
pytest tests/models -m slow      # when changing a model wrapper (needs GPU)
pytest tests/ -m "not slow"      # CI default
```

## Layout

```
hanoi_caption/        importable package (video_pipeline, dam_model, kb_loader, model_registry, schemas)
scripts/eval/         evaluation scripts (see above)
scripts/data_collection/   FeatureExtractor + KB image crawlers
notebooks/            video pipeline demo
data/kb.json          KB export
data/kb_images/       reference images per landmark (gitignored)
data/cache/           embeddings + HF model cache (gitignored)
data/eval/            eval inputs, predictions, metrics, summaries
tests/                pytest suite (see above)
```
