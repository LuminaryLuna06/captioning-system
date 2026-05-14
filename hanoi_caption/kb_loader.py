"""Load the KB JSON file into a dict of KBNode.

The KB on disk uses unsuffixed English field names (``name``, ``description``,
``visual_cues``); the schema still uses the historical ``_en`` variants. The
adapter below renames before validation so downstream code is unchanged.
Pydantic v2 ignores the KB's other metadata fields (kb_id, region, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

from hanoi_caption.schemas import KBNode

_LEGACY_FIELD_RENAMES = (
    ("name", "name_en"),
    ("description", "description_en"),
    ("visual_cues", "visual_cues_en"),
)


def _normalize_legacy_fields(item: dict) -> dict:
    out = dict(item)
    for new_key, old_key in _LEGACY_FIELD_RENAMES:
        if new_key in out and old_key not in out:
            out[old_key] = out.pop(new_key)
    return out


def load_kb(path: Path | str, only_objects: bool = True) -> dict[str, KBNode]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [KBNode.model_validate(_normalize_legacy_fields(item)) for item in raw]
    if only_objects:
        nodes = [n for n in nodes if n.type == "object"]
    return {n.id: n for n in nodes}


def index_by_kb_id(nodes: dict[str, KBNode]) -> dict[str, KBNode]:
    """Reindex KB nodes by their human-readable ``kb_id`` slug.

    The crawler writes images under ``data/kb_images/<kb_id>/``, so retrievers
    that recover a node id from a path need this slug-keyed lookup rather than
    the opaque ``id``.
    """
    return {n.kb_id: n for n in nodes.values() if n.kb_id}
