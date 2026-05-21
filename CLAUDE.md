# CLAUDE.md

Hanoi landmark captioning system. KB-grounded captions (150–300 words, tour-guide English) for images and videos.

## Environment

- Python: `D:\Jupiter\luna_env\Scripts\python.exe` (luna_env, PyTorch + CUDA 12.8 for Blackwell sm_120). Base Anaconda Python has a DLL conflict — do not use it.

## Pipelines

### Image — `hanoi_caption/pipeline_retriever.py`

1. **Describe** — VLM (Qwen) produces a holistic description of the image.
2. **Match** — BGE-M3 cosine retrieval over KB `visual_cues` → top-K candidates → VLM rerank picks one landmark (or refuses with `"Not a recognized Hanoi landmark."`).
3. **Caption** — DAM-3B writes the final paragraph with the matched KB node's facts (`name_en`, `description_en`, `visual_cues_en`) embedded in its prompt.

### Video — `hanoi_caption/video_pipeline.py`

1. **Sample** — decode frames at `sample_fps` (default 1.0) via OpenCV.
2. **Retrieve per frame** — DINOv3 features → FAISS cosine top-1 against a pre-built reference-image index → `(kb_id, score)`. Index + `id_map.json` live under `data/cache/`.
3. **Smooth & group** — sliding-window majority vote over kb_ids, merge consecutive same-id frames into runs, absorb short runs into the longer known neighbor, drop unknown segments shorter than `min_segment_seconds`.
4. **Caption per segment** — `pick_frame_indices` evenly samples K frames within `dam_frame_budget=(4,8)`; DAM-3B receives all K frames + all-ones masks and writes one KB-grounded paragraph naming the landmark.

Public entrypoint: `caption_video(video_path, kb_nodes, dino_index_path, id_map_path, ...)` → `list[VideoSegment]`.

## Knowledge base

- `data/kb.json` — bilingual KB nodes (`kb_id`, `name_en`, `description_en`, `visual_cues_en`, ...).
- `data/kb_images/<kb_id>/*.jpg` — reference images per landmark (gitignored). Used to build the DINOv3 index.
- `data/cache/kb_index_*.npz`, `data/cache/id_map.json` — built artifacts (gitignored).

## Notebooks

- `notebooks/01_phase1_kb_only.ipynb` — KB-only smoke test.
- `notebooks/02_phase2_full_pipeline.ipynb` — full image pipeline + timing comparison.

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

## Layout

```
hanoi_caption/        importable package (pipeline_retriever, video_pipeline, kb_*, schemas, model_registry, ...)
scripts/eval/         evaluation scripts (see above)
scripts/data_collection/   FeatureExtractor + KB image crawlers
notebooks/            phase 1 and phase 2 notebooks
data/kb.json          KB export
data/kb_images/       reference images per landmark (gitignored)
data/cache/           embeddings + HF model cache (gitignored)
data/eval/            eval inputs, predictions, metrics, summaries
tests/                pytest suite + tests/videos/ fixtures (gitignored)
```
