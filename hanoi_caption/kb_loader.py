"""Load the KB JSON file into a dict of KBNode."""
from __future__ import annotations

import json
from pathlib import Path

from hanoi_caption.schemas import KBNode


def load_kb(path: Path | str, only_objects: bool = True) -> dict[str, KBNode]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = [KBNode.model_validate(item) for item in raw]
    if only_objects:
        nodes = [n for n in nodes if n.type == "object"]
    return {n.id: n for n in nodes}
