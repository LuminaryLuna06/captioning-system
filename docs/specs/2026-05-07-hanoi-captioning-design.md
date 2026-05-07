# Hanoi Travel Image Captioning System — Baseline Design

- **Date:** 2026-05-07
- **Scope:** v1 baseline. Image-only. Landmarks only. English captions, 150–300 words, tour-guide style.
- **Out of scope (v1):** Food, street life, video, Vietnamese-language output, formal evaluation.

## 1. Goal

Given an image of a Hanoi landmark, produce a 150–300 word English tour-guide-style caption that combines:

1. **Vetted facts** from a manually-built bilingual knowledge base (KB) of Hanoi landmarks.
2. **Visual specifics** actually present in the input photo (this angle, this lighting, these foreground details).

Two photos of the same landmark must produce visibly different captions — that is the difference between this system and "look up the Wikipedia entry for the landmark."

## 2. Core Idea

The KB is the spine of the system. It does two jobs:

1. **Identifies the landmark** — by semantic-matching the image's visual content against each KB node's `Visual Cues` field.
2. **Directs visual attention** — its `Visual Cues` are converted into detection queries that tell the region-describer (DAM) what to look at.

A composer LLM then weaves KB facts together with DAM's region-level descriptions into the final paragraph.

## 3. Pipeline

### 3.1 Data flow (online, per image)

```
Input image
     │
     ├─▶ [3] Whole-image describer (Qwen2.5-VL-7B)
     │     "Describe what is visible. No landmark names."
     │     → holistic_desc
     │
     ├─▶ [4] KB matcher
     │     - Embed holistic_desc with BGE-M3
     │     - Cosine vs. cached KB Visual-Cue embeddings → top-3 candidates
     │     - VLM re-rank: image + 3 candidates → final node | "none"
     │     ↓
     │   Decision gate: confidence ≥ MATCH_THRESHOLD AND re-rank ≠ "none"?
     │     - No  → return refusal: "Not a recognized Hanoi landmark." STOP.
     │     - Yes → matched_node, continue.
     │
     ├─▶ [5] Query extractor (Qwen2.5-7B-Instruct)
     │     matched_node.visual_cues_en → JSON list of 4–8 short noun-phrase queries
     │     e.g. ["stone stele", "turtle pedestal", "tiered red roof"]
     │
     ├─▶ [6] Region proposer
     │     - Grounding DINO (text-prompted detection, queries from step 5)
     │     - SAM 2 (boxes → masks)
     │     - Filter: drop masks <1% area, drop IoU > 0.7 overlaps, cap at top 6 by score
     │     - Fallback: if zero detections, run SAM 2 auto-mask, take top 4 by area
     │
     ├─▶ [7] Region describer (DAM-3B)
     │     For each (image, mask) → 1–2 sentence focal description
     │     → list[RegionDescription]
     │
     └─▶ [8] Composer (Qwen2.5-7B-Instruct, text-only)
           Prompt:
             SYSTEM: tour-guide voice, 150–300 words, prose not list,
                     no facts beyond what is provided.
             USER: landmark name + KB description_en
                   + bullet list of region descriptions
                   + holistic_desc
           → final caption (150–300 words)
```

### 3.2 Stage purposes

| Stage | Purpose |
|---|---|
| Whole-image describer | Produce text in the same descriptive space as the KB's Visual Cues field, so cosine similarity is meaningful. Forbidden from naming landmarks to keep it task-focused. |
| KB matcher | Pick the right KB node (or refuse). Two-stage (cheap embedding retrieval + costlier VLM re-rank) for speed and accuracy. |
| Query extractor | Translate the KB node's prose Visual Cues into terse, detector-friendly noun phrases. |
| Region proposer | Locate those things in *this specific* image, produce per-object masks. |
| Region describer (DAM) | The system's actual research contribution — fine-grained focal descriptions of masked regions. |
| Composer | Blend KB facts (the "what is this place") with DAM region details (the "what's actually in this photo") into one paragraph. |

### 3.3 Latency budget (single image, single 16 GB GPU)

| Stage | Time |
|---|---|
| Whole-image describe | ~1.5 s |
| Embed + cosine + VLM re-rank | ~1 s |
| Query extraction | ~0.5 s |
| Detect + segment + filter | ~1.5 s |
| DAM × ~6 regions | ~3 s |
| Compose | ~3 s |
| Sequential model swap overhead (Path A) | ~3–5 s |
| **Total** | **~14–15 s/image** |

## 4. Components

Each component is a Python module with a small, testable interface. The notebook imports them and runs cells.

| # | Module | Model | Input → Output |
|---|---|---|---|
| 1 | `kb_loader` | — | Path to KB JSON → `dict[node_id → KBNode]` (Pydantic) |
| 2 | `kb_indexer` (offline) | BGE-M3 | KB nodes → `(node_ids, np.ndarray of Visual-Cue embeddings)`, cached to `.npz` |
| 3 | `image_describer` | Qwen2.5-VL-7B-Instruct | `PIL.Image` → holistic visual description (str) |
| 4 | `kb_matcher` | BGE-M3 + Qwen2.5-VL-7B (re-rank) | `(holistic_desc, image, kb_index, threshold)` → `MatchResult { node_id \| None, confidence }` |
| 5 | `query_extractor` | Qwen2.5-7B-Instruct | Visual Cues text → `list[str]` of detection queries |
| 6 | `region_proposer` | Grounding DINO + SAM 2 | `(image, queries)` → `list[Region { box, mask, query, score }]` |
| 7 | `region_describer` | DAM-3B | `(image, mask)` per region → `list[RegionDescription]` |
| 8 | `composer` | Qwen2.5-7B-Instruct | `(kb_node, region_descriptions, holistic_desc)` → caption (str) |
| 9 | `pipeline` | — | Orchestrates 3 → 4 → 5 → 6 → 7 → 8. Returns `CaptionResult`. |

**Module dependency graph:** strict tree, no cycles.

```
pipeline ──▶ image_describer
         ──▶ kb_matcher ──▶ kb_indexer ──▶ kb_loader
         ──▶ query_extractor
         ──▶ region_proposer
         ──▶ region_describer
         ──▶ composer
```

**Cross-cutting:**
- All inter-module values are Pydantic models.
- `model_registry.py` is the single source of truth for model loading. It implements lazy-load and (for Path A) sequential evict/load to keep VRAM under 14 GB.
- Each pipeline run emits a `debug` dict carrying every intermediate value (top-k matches, scores, masks as PNG bytes, region descriptions, full composer prompt). The notebook renders this. **This is the v1 evaluation surface** — the user opted out of formal eval.

## 5. KB Schema & Integration

### 5.1 Node schema (consumed by the pipeline)

```python
class KBNode(BaseModel):
    id: str
    name_en: str
    name_vi: str
    type: Literal["object", "category"]
    parent_id: str | None
    description_en: str       # tour-guide narrative (used by composer)
    description_vi: str       # not used in v1
    visual_cues_en: str       # used for matching + query extraction
    visual_cues_vi: str       # not used in v1
    tags: list[str]           # not used in v1
```

Only nodes with `type == "object"` are indexed. Category nodes (e.g., `categoryHaNoi`) are skipped.

### 5.2 Field usage

| Field | Used by | Purpose |
|---|---|---|
| `name_en` | composer, kb_matcher (re-rank prompt) | The landmark's name in the prompt |
| `description_en` | composer | Factual content the caption weaves in |
| `visual_cues_en` | kb_indexer | Embedded for cosine matching |
| `visual_cues_en` | query_extractor | Source for detection queries |
| All others | — | Reserved for v2+ |

### 5.3 KB JSON shape

```json
[
  {
    "id": "temple_of_literature",
    "name_en": "Temple of Literature",
    "name_vi": "Văn Miếu – Quốc Tử Giám",
    "type": "object",
    "parent_id": "categoryHaNoi",
    "description_en": "...",
    "description_vi": "...",
    "visual_cues_en": "...",
    "visual_cues_vi": "...",
    "tags": []
  }
]
```

A 15-node Gemini-generated sample lives at `data/kb.json` for development. The user's real KB tool exports to this same shape (export wiring is a downstream task, tracked outside this spec).

### 5.4 Visual Cues authoring guidelines

Two rules the human KB authors must follow — retrieval and detection quality depend on them.

1. **Be visually concrete, not poetic.** "Five-tier red-tiled roof curving upward at the eaves, supported by carved wooden brackets" beats "a roof reaching toward heaven."
2. **Use detector-friendly nouns.** Include common-vocabulary objects: stele, turtle pedestal, gate, courtyard, lotus pond, tiered roof, dragon carving, white tower, Gothic spire, etc.

### 5.5 Indexing strategy

- KB is small (<100 nodes for v1) → in-memory `numpy` array, cosine via dot product. No FAISS, no vector DB.
- Cache `(node_ids, embeddings)` to `data/cache/kb_index.npz` keyed by KB file content hash.
- Re-embed automatically on KB change.

## 6. Decision Gates & Failure Paths

| Condition | Action |
|---|---|
| Top-1 cosine < `MATCH_THRESHOLD` (start at 0.45) | Refuse |
| VLM re-rank returns "none" | Refuse |
| Otherwise | Proceed with matched node |
| Zero Grounding DINO detections survive filtering | Fall back to SAM 2 auto-mask, top 4 by area |
| DAM produces near-identical region descriptions | (Future) text-level dedup; v1 accepts as-is |
| Composer drifts from tour-guide tone | (Future) add 1–2 in-context examples; v1 starts with system prompt only |

The refusal path returns `CaptionResult { caption=None, refusal="Not a recognized Hanoi landmark." }`. No best-guess fallback.

## 7. Phasing

### 7.1 Phase 1 — KB-only smoke test (~1 day)

**Goal:** prove the KB → identify → compose path end-to-end before adding DAM.

Implements: `kb_loader`, `kb_indexer`, `image_describer`, `kb_matcher`, `composer`.
Skips: `query_extractor`, `region_proposer`, `region_describer`.

Pipeline collapses to: image → holistic_desc → KB match → composer (KB description + holistic_desc; no region descriptions).

**Validates:**
- KB JSON loads and embeds correctly.
- Visual-Cue similarity retrieves the right node on hand-picked images.
- VLM re-rank disambiguates similar-looking landmarks (multiple pagodas, multiple temples).
- Composer hits the tone and 150–300 word target.
- Refusal path fires on out-of-KB images.

**Expected limitation:** captions for two photos of the same landmark are near-identical. This is correct for Phase 1 and is the motivation for Phase 2.

### 7.2 Phase 2 — Add DAM region grounding (~2–3 days)

Adds: `query_extractor`, `region_proposer`, `region_describer`. Plugs into the existing pipeline.

**Validates:** three different photos of the same landmark produce three visibly different captions, each grounded in its own visual specifics.

**New risks introduced:**
- Detector misses on stylized cues → SAM auto-mask fallback covers gaps.
- DAM repetitive output across overlapping masks → IoU filter is first defense; text dedup if needed.
- Composer ignores KB facts in favor of region details → tunable via prompt structure.

### 7.3 Phase 3 — Video extension (out of scope for this spec, sketched only)

Reuses everything from Phase 2. Only the region-description stage changes: SAM 2 mask propagation tracks Phase 2's keyframe masks across the clip; DAM-Video produces temporal region descriptions; composer absorbs motion language. SAM 2 and DAM were chosen partly because both have native video variants — Phase 3 should be a small extension, not a rewrite.

## 8. Project Layout

```
CaptioningSystem/
├── pyproject.toml
├── README.md
├── data/
│   ├── kb.json                    # the KB export (replaceable)
│   └── cache/
│       ├── kb_index.npz           # cached BGE-M3 embeddings
│       └── hf_models/             # HF model cache
├── notebooks/
│   └── 01_pipeline.ipynb          # the actual UI
├── hanoi_caption/                 # importable package
│   ├── __init__.py
│   ├── schemas.py                 # Pydantic models
│   ├── kb_loader.py
│   ├── kb_indexer.py
│   ├── image_describer.py
│   ├── kb_matcher.py
│   ├── query_extractor.py
│   ├── region_proposer.py
│   ├── region_describer.py
│   ├── composer.py
│   ├── pipeline.py
│   └── model_registry.py          # lazy-load + sequential eviction
├── docs/
│   └── specs/
│       └── 2026-05-07-hanoi-captioning-design.md
└── tests/
    └── fixtures/                  # 5–10 test images for smoke runs
```

The notebook is glue + display only. All real logic lives in `hanoi_caption/`.

## 9. Dependencies

Python ≥ 3.10.

```
torch>=2.6                # or PyTorch nightly with CUDA 12.8+ for Blackwell sm_120
transformers>=4.46
accelerate>=1.0
bitsandbytes>=0.45
sentencepiece
pillow
numpy
pydantic>=2
huggingface_hub
einops

# Vision/segmentation
groundingdino-py
sam2

# Embeddings
FlagEmbedding             # BGE-M3

# DAM (NVIDIA, install from GitHub)
# pip install git+https://github.com/NVlabs/describe-anything.git

# Notebook
jupyterlab
ipykernel
matplotlib
```

## 10. Hardware Plan: Path A on 16 GB VRAM (RTX 5060 Ti, Blackwell sm_120)

### 10.1 Sequential model loading

The full stack at ~19 GB does not fit concurrently. `model_registry` is responsible for keeping the loaded working set under ~14 GB. It tracks individual model handles and evicts on demand, rather than thinking in fixed groups. The expected timeline for a single image:

| Pipeline stage | Models added | Models evicted | Peak VRAM |
|---|---|---|---|
| 3 — Whole-image describe | + Qwen2.5-VL-7B, + BGE-M3 | — | ~7 GB |
| 4 — KB match (cosine + re-rank) | — | — | ~7 GB |
| 5 — Query extract | + Qwen2.5-7B-Instruct | — | ~12 GB |
| 6 — Detect + segment | + Grounding DINO, + SAM 2 | − Qwen2.5-VL-7B | ~10 GB |
| 7 — Region describe | + DAM-3B | − Grounding DINO, − SAM 2 (after step 6) | ~9 GB |
| 8 — Compose | — | − DAM-3B, − BGE-M3 | ~5 GB |

Note that **Qwen2.5-7B-Instruct is shared between stage 5 (query extraction) and stage 8 (composition)** and stays resident across the run — both stages are short, text-only inferences on the same weights with different prompts. This eliminates the most expensive potential reload.

Peak VRAM stays at ~12 GB, leaving margin for activation memory and KV cache. If profiling shows we exceed 14 GB on real images, the first lever is to evict Qwen2.5-VL-7B immediately after stage 4 instead of waiting for stage 6.

### 10.2 Blackwell-specific risks

`sm_120` is brand-new silicon as of the design date. Stable PyTorch wheels do not ship sm_120 kernels. Required:

1. PyTorch nightly with CUDA 12.8+, OR PyTorch 2.6+ stable when available.
2. `bitsandbytes ≥ 0.45` for 4-bit quant on Blackwell.
3. SAM 2 and Grounding DINO have custom CUDA kernels — likely need rebuild from source against the chosen torch.
4. DAM is a transformers-based model with no custom kernels; should work as-is once torch is right.

**Implementation plan must allocate 1–2 days for environment setup before any captioning code runs.** If a kernel build fails on sm_120 and blocks progress, the documented fallback is to route the affected module to an API and continue.

### 10.3 4-bit quantization

All 7B-class models load in 4-bit (NF4 via bitsandbytes) by default. DAM-3B in fp16. SAM 2 and Grounding DINO in fp16. This is what makes the per-group VRAM budget realistic.

## 11. Risks (Rank-Ordered)

1. **Visual Cues authoring quality.** The whole identification path leans on this field being concrete and detection-friendly. Mitigation: enforce the Section 5.4 guidelines in the KB authoring tool itself.
2. **Match threshold tuning.** `MATCH_THRESHOLD` has no principled value. Plan: start at cosine 0.45 + require VLM re-rank ≠ "none". Tune after eyeballing 10–20 outputs.
3. **Blackwell environment setup.** Custom-kernel components may need rebuilds. Budget 1–2 days. Fallback: route affected module to API.
4. **Detector miss on stylized cues.** Mitigation: query extractor favors common vocabulary; SAM-auto-mask fallback.
5. **DAM hallucination on unusual masks.** Mitigation: small-mask filter; composer prompt forbids invented facts (but may parrot DAM — eyeball during dev).
6. **Tone consistency.** Composer may drift to Wikipedia voice. Mitigation: add 1–2 in-context examples once a few liked outputs exist.
7. **KB scope creep.** Resist adding food/streets to v1.

## 12. Out of Scope for v1

- Vietnamese-language caption output (KB is bilingual; we use the EN side only).
- Food, street life, festivals, interiors, signage.
- Formal evaluation (BLEU/ROUGE, LLM-as-judge, labeled test sets).
- API or service deployment (notebook only).
- Best-guess captioning when KB has no match (we refuse instead).
- Caching captions for repeated images.
- KB authoring tooling.
- KB JSON export wiring from the existing KB UI (handled separately by user).

## 13. Open Tasks Outside This Spec

- User wires up KB JSON export from their existing KB editor to the schema in §5.3.
- User provides a small fixture set of 5–10 Hanoi landmark photos for `tests/fixtures/`, plus 2–3 non-Hanoi or non-landmark photos to validate the refusal path.

## 14. Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| Use case | Tour-guide content generation | User pick (Q1=C) |
| Caption length/language | English, 150–300 words | User pick (Q2=D) |
| Input scope | Landmarks only | User pick (Q3=E narrow → Q3'=A) |
| KB source | Manually-built bilingual KB (existing) | User pick (Q4) |
| Identification | Visual-Cues semantic matching + VLM re-rank | User pick (Q5=C); reuses existing KB field, no reference-image curation |
| DAM role | Region grounding for final caption | User pick (Q6=B); plays to DAM's strength |
| Mask source | KB-driven detection (Grounding DINO + SAM 2), SAM-auto-mask fallback | User pick (Q7=B) |
| Composer | Open-weight LLM, not DAM | User pick (Q8=D); DAM is a focal describer, not a composer |
| Out-of-KB | Refuse | User pick (Q9=A) |
| Eval | None for v1 | User pick (Q10=E) |
| Deployment | Jupyter notebook | User pick (Q11=A) |
| Hardware path | Path A (fully local, sequential loading) | User pick; user has RTX 5060 Ti 16 GB |
