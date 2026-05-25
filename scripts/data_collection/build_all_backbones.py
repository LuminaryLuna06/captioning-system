"""Build the FAISS retrieval cache once per backbone.

    python scripts/data_collection/build_all_backbones.py
        [--backbones dinov3,resnet50,siglip2,vit]   # default: all four
        [--kb-dir data/kb_images]
        [--cache-dir data/cache]
        [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hanoi_caption.retrieval.backbones import (  # noqa: E402
    Dinov3Extractor,
    Resnet50Extractor,
    Siglip2Extractor,
    VitExtractor,
)
from hanoi_caption.retrieval.index import build_or_load_index  # noqa: E402

REGISTRY = {
    "dinov3":   Dinov3Extractor,
    "resnet50": Resnet50Extractor,
    "siglip2":  Siglip2Extractor,
    "vit":      VitExtractor,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbones", default=",".join(REGISTRY.keys()),
                   help="Comma-separated subset of: " + ",".join(REGISTRY.keys()))
    p.add_argument("--kb-dir", default="data/kb_images", type=Path)
    p.add_argument("--cache-dir", default="data/cache", type=Path)
    p.add_argument("--force", action="store_true", help="Rebuild even if cache exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("build_all_backbones")

    requested = [b.strip() for b in args.backbones.split(",") if b.strip()]
    unknown = [b for b in requested if b not in REGISTRY]
    if unknown:
        log.error("unknown backbones: %s (valid: %s)", unknown, list(REGISTRY))
        return 2

    for name in requested:
        log.info("=== %s ===", name)
        t0 = time.perf_counter()
        ext = REGISTRY[name]()
        index, id_map = build_or_load_index(
            ext, args.kb_dir, args.cache_dir, force_rebuild=args.force,
        )
        log.info("%s ready: %d vectors, dim=%d, %.1fs",
                 name, index.ntotal, ext.dim, time.perf_counter() - t0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
