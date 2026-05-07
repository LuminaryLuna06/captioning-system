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
