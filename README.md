# CaptioningSystem

Baseline KB-grounded image captioning for Hanoi landmarks. Combines a manually-built bilingual knowledge base with the Describe Anything Model (DAM) to produce 150–300 word tour-guide-style English captions grounded in both vetted facts and per-photo visual specifics.

## Architecture: Retriever Pipeline

The current pipeline uses a **retriever pipeline** architecture to match images to knowledge base (KB) entries and generate captions:

1. **Initial Image Description:** The Describe Anything Model (DAM) is used to scan the input image and generate an initial, detailed visual description (replacing the Qwen model in this step).
2. **KB Matching & Reranking:** The visual description is used to retrieve candidate landmarks from the Knowledge Base. The Qwen model is then used to rerank the **top-5 candidates** to definitively identify the landmark or reject if there's no match.
3. **Caption Composition:** Based on the identified KB node and visual specifics, the final tour-guide-style caption is generated.

## Proposed Future Architecture: Visual RAG (DINOv3 + DAM)

To address current latency bottlenecks and VRAM constraints (model swapping between Qwen, DAM, and BGE), the system is proposed to migrate to a **Visual RAG** architecture using Meta's DINOv3 as the vision encoder.

**Proposed Pipeline:**
1. **Image-to-Image Retrieval (DINOv3):** Instead of generating text to search the KB, the input image is passed through DINOv3 to extract a feature vector (~10-30ms).
2. **Vector Search:** This vector is compared via Cosine Similarity against a pre-computed vector database of "reference images" for each landmark in the KB. This definitively identifies the landmark with high accuracy and near-zero latency.
3. **Generation (DAM-3B):** The identified landmark's historical context is injected into DAM-3B's prompt, which then generates the final tour-guide caption based on the image and provided facts.

**Key Benefits:**
- **Real-time Latency:** Bypasses the slow autoregressive text generation of Qwen during the retrieval phase.
- **VRAM Efficiency:** DINO (small ViT) and DAM-3B can reside in memory concurrently on a 16GB GPU without swapping.
- **Robustness:** Eliminates the dependency on human-authored, exact-match text `visual_cues`. Replaces textual cues with 3-5 reference images per landmark.

## Setup (RTX 5060 Ti / Blackwell)

This project uses the existing `luna_env` environment (configured with PyTorch + CUDA 12.8 for Blackwell sm_120). 

```bash
conda activate luna_env   # or: mamba activate luna_env
pip install -e ".[dev]"
pip install groundingdino-py sam2
pip install git+https://github.com/NVlabs/describe-anything.git
```

## Run

```bash
jupyter lab notebooks/01_phase1_kb_only.ipynb
jupyter lab notebooks/02_phase2_full_pipeline.ipynb
```

The `02_phase2_full_pipeline.ipynb` notebook includes batch processing capabilities to run the retriever pipeline sequentially on all test photos, complete with a matplotlib timing comparison table.

## Project layout

```
.
├── pyproject.toml
├── data/
│   ├── kb.json                  # KB export (replaceable)
│   └── cache/                   # gitignored: embeddings + HF model cache
├── notebooks/
│   ├── 01_phase1_kb_only.ipynb     # Phase 1 smoke test (no DAM)
│   └── 02_phase2_full_pipeline.ipynb  # Phase 2 full pipeline + comparison
├── hanoi_caption/               # importable package
├── docs/                        # design specs and plans
└── tests/
    └── fixtures/                # user-supplied test images
```
