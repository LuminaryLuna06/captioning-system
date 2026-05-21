"""DAM-3B model registration.

DAM is installed from NVIDIA's repo:
    pip install git+https://github.com/NVlabs/describe-anything.git

The shipped `dam` package exposes `DescribeAnythingModel`, takes a *local
checkpoint dir* (not a HF model id) for `model_path`, requires explicit
`conv_mode` + `prompt_mode` constructor args, and offers
`get_description(image_pil, mask_pil, query=...)`. This module wraps that
real API and registers a lazy loader with the model registry so the video
pipeline can fetch DAM on demand.
"""
from __future__ import annotations

from hanoi_caption.model_registry import registry

MODEL_NAME = "dam_3b"
DAM_REPO = "nvidia/DAM-3B"
DAM_CONV_MODE = "v1"
DAM_PROMPT_MODE = "full+focal_crop"


def _load():
    # --- compatibility shims for transformers ≥5.0 ↔ dam ≤current ---
    # 1. transformers 5.x removed `no_init_weights` from `modeling_utils`;
    #    the installed `dam` package still imports it from the old path.
    # 2. transformers 5.x added `mark_tied_weights_as_initialized` that
    #    iterates `self.all_tied_weights_keys.keys()`, but DAM-bundled
    #    sub-models (e.g. MultimodalProjector) only define the older
    #    `_tied_weights_keys` list. Default the new attribute to {} so the
    #    iteration is a no-op — tied-weights bookkeeping is diagnostic only
    #    and does not affect inference correctness.
    import transformers.modeling_utils as _mu
    if not hasattr(_mu, "no_init_weights"):
        from transformers.initialization import no_init_weights as _niw
        _mu.no_init_weights = _niw
    if not hasattr(_mu.PreTrainedModel, "all_tied_weights_keys"):
        _mu.PreTrainedModel.all_tied_weights_keys = {}

    from huggingface_hub import snapshot_download
    from dam import DescribeAnythingModel  # type: ignore

    # DAM expects a *local* checkpoint dir. snapshot_download (no local_dir)
    # places the snapshot in the standard HF cache and returns its path —
    # no extra disk copy.
    model_path = snapshot_download(DAM_REPO)

    model = DescribeAnythingModel(
        model_path=model_path,
        conv_mode=DAM_CONV_MODE,
        prompt_mode=DAM_PROMPT_MODE,
    )
    model.eval()
    model.to("cuda")
    return model


registry.register(MODEL_NAME, _load)
