# CaptioningSystem

KB-grounded **video** captioning for Hanoi landmarks. For each segment of a walking-tour clip, the system identifies the landmark via DINOv3 image retrieval and writes a 150–300 word tour-guide paragraph with DAM-3B, grounded in a hand-curated bilingual knowledge base.

## Pipeline

`hanoi_caption/video_pipeline.py` — `caption_video()`:

1. **Sample frames** at `sample_fps` (default 1.0 fps) via OpenCV.
2. **Retrieve per frame** — DINOv3 features → FAISS cosine top-1 against a pre-built reference-image index → `(kb_id, score)`.
3. **Smooth & group** — sliding-window majority vote over kb_ids, merge consecutive same-id frames into runs, absorb short runs into the longer known neighbor, drop unknown segments shorter than `min_segment_seconds`.
4. **Caption per segment** — `pick_frame_indices` evenly samples K frames within `dam_frame_budget=(4,8)`; DAM-3B receives all K frames + all-ones masks and writes one KB-grounded paragraph naming the landmark.

Returns `list[VideoSegment]` (start_s, end_s, kb_id, name_en, confidence, caption, debug).

## Setup (RTX 5060 Ti / Blackwell)

```bash
conda activate luna_env   # PyTorch + CUDA 12.8 (sm_120)
pip install -e ".[dev]"
pip install git+https://github.com/NVlabs/describe-anything.git
```

## Run

```python
from hanoi_caption.kb_loader import load_kb
from hanoi_caption.video_pipeline import caption_video

nodes = load_kb("data/kb.json")
segments = caption_video(
    video_path="tests/videos/my_clip.mp4",
    kb_nodes=nodes,
    dino_index_path="data/cache/dino_faiss.index",
    id_map_path="data/cache/id_map.json",
    sample_fps=1.0,
    smooth_window=3,
    min_segment_seconds=2.0,
    dam_frame_budget=(4, 8),
)
```

Or use `notebooks/02_phase2_full_pipeline.ipynb` for an interactive demo with a timeline plot.

## Knowledge base

- `data/kb.json` — bilingual KB nodes (`kb_id`, `name_en`, `description_en`, `visual_cues_en`, …).
- `data/kb_images/<kb_id>/*.jpg` — reference images per landmark (gitignored). Used to build the DINOv3 + FAISS index via `scripts/data_collection/`.
- `data/cache/dino_faiss.index`, `data/cache/id_map.json` — built artifacts (gitignored).

## Evaluation

`scripts/eval/` — run in order:

1. `generate_landmark_map.py` → `data/eval/landmark_map.json`
2. `build_test_set.py --video-dir tests/videos` → `data/eval/test_set.json`
3. `run_pipeline.py` → `data/eval/pipeline_results.json`
4. `eval_segmentation.py` → `data/eval/seg_metrics.json`
5. `eval_caption.py` → `data/eval/caption_metrics.json`
6. `eval_llm_judge.py` → `data/eval/llm_scores.json`
7. `eval_summary.py` → `data/eval/eval_summary.{csv,tex}`

Per-experiment snapshots go under `data/eval/runs/<date>_<label>/`.

## Layout

```
hanoi_caption/             importable package
  video_pipeline.py        caption_video() entrypoint
  dam_model.py             DAM-3B loader + registry registration
  kb_loader.py             load_kb() and index_by_kb_id()
  model_registry.py        lazy model loading + LRU eviction
  schemas.py               KBNode, VideoSegment
scripts/eval/              eval pipeline (segmentation, caption, LLM judge, summary)
scripts/data_collection/   FeatureExtractor + KB image crawlers
notebooks/                 interactive demo
data/kb.json               KB export
tests/                     pytest suite
```
