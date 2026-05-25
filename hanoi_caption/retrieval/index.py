"""Build-or-load FAISS index per backbone, cached under {cache_dir}/{extractor.name}/."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError

log = logging.getLogger(__name__)

_VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(kb_dir: Path) -> list[Path]:
    return sorted(p for p in kb_dir.rglob("*") if p.suffix.lower() in _VALID_EXT)


def build_or_load_index(
    extractor,
    kb_images_dir: Path | str,
    cache_dir: Path | str = "data/cache",
    batch_size: int = 16,
    force_rebuild: bool = False,
) -> Tuple[faiss.Index, dict[int, str]]:
    """Return (faiss_index, id_map). Build if cache missing, else load."""
    kb_dir = Path(kb_images_dir)
    cache_path = Path(cache_dir) / extractor.name
    index_path = cache_path / "faiss.index"
    map_path = cache_path / "id_map.json"

    if not force_rebuild and index_path.exists() and map_path.exists():
        log.info("Loading cached index for %s from %s", extractor.name, cache_path)
        index = faiss.read_index(str(index_path))
        with open(map_path, "r", encoding="utf-8") as f:
            id_map = {int(k): v for k, v in json.load(f).items()}
        return index, id_map

    image_paths = _list_images(kb_dir)
    if not image_paths:
        raise ValueError(f"no images found under {kb_dir}")

    cache_path.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(extractor.dim)
    id_map: dict[int, str] = {}

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        valid_paths = []
        for p in batch_paths:
            try:
                with Image.open(p) as im:
                    batch_images.append(im.convert("RGB"))
                valid_paths.append(p)
            except (UnidentifiedImageError, OSError) as e:
                log.warning("skipping unreadable image %s: %s", p, e)
        if not batch_images:
            continue
        embeddings = extractor.extract(batch_images).astype("float32")
        start_id = index.ntotal
        for j, p in enumerate(valid_paths):
            id_map[start_id + j] = str(p.resolve())
        index.add(embeddings)
        log.info("indexed %d/%d images for %s", index.ntotal, len(image_paths), extractor.name)

    faiss.write_index(index, str(index_path))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
    return index, id_map
