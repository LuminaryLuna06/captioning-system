# CaptioningSystem

Baseline KB-grounded image captioning for Hanoi landmarks. Combines a manually-built bilingual knowledge base with the Describe Anything Model (DAM) to produce 150–300 word tour-guide-style English captions grounded in both vetted facts and per-photo visual specifics.

- **Design spec:** [`docs/specs/2026-05-07-hanoi-captioning-design.md`](docs/specs/2026-05-07-hanoi-captioning-design.md)
- **Implementation plan:** [`docs/plans/2026-05-07-hanoi-captioning.md`](docs/plans/2026-05-07-hanoi-captioning.md)

## Resuming work in a new Claude Code session

This project was scaffolded in a Mac session and is being handed off to a session on the GPU machine. To pick up where it left off:

1. **Read in this order, no exceptions:**
   1. This README (top-to-bottom).
   2. `docs/specs/2026-05-07-hanoi-captioning-design.md` — the full design.
   3. `docs/plans/2026-05-07-hanoi-captioning.md` — the task-by-task plan with code blocks.

2. **Check what's already done:** run `git log --oneline` to see committed tasks. As of the handoff, **Task 0.1 (project scaffold) is committed**; Tasks 0.2 onward are pending.

3. **Resume execution at Task 0.2** in `docs/plans/...`. The plan was being executed via the `superpowers:subagent-driven-development` skill — invoke it again to continue, or execute tasks inline.

4. **Sample KB:** an existing 15-node Gemini-generated sample KB lives at `~/hanoi_kb_sample.json` on the original Mac. It is **not** in this repo. Either:
   - Copy it to `data/kb.json` (Task 0.3 in the plan), or
   - Regenerate it on the GPU PC by running the same Gemini CLI prompt from the brainstorming session, or
   - Wait for the user's real KB export (the user said this would come later).

5. **Hardware/environment note:** the plan targets RTX 5060 Ti 16 GB (Blackwell sm_120). Task 0.2 installs PyTorch nightly with CUDA 12.8. If `torch.cuda.get_device_capability(0)` does not return `(12, 0)` after Task 0.2 Step 3, **stop and resolve before continuing** — most other tasks depend on this.

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

## Project layout (target — most files are added by the plan)

```
.
├── pyproject.toml
├── data/
│   ├── kb.json                  # KB export (replaceable)
│   └── cache/                   # gitignored: embeddings + HF model cache
├── notebooks/
│   ├── 01_phase1_kb_only.ipynb     # Phase 1 smoke test (no DAM)
│   └── 02_phase2_full_pipeline.ipynb  # Phase 2 full pipeline + comparison
├── hanoi_caption/               # importable package (stages 3-8)
├── docs/
│   ├── specs/2026-05-07-hanoi-captioning-design.md
│   └── plans/2026-05-07-hanoi-captioning.md
└── tests/
    └── fixtures/                # 5-10 user-supplied test images
```
