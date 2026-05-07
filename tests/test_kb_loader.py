# tests/test_kb_loader.py
from pathlib import Path
import pytest

from hanoi_caption.kb_loader import load_kb


def test_load_kb_returns_dict_keyed_by_id(tmp_path: Path):
    kb_file = tmp_path / "kb.json"
    kb_file.write_text(
        '[{"id":"a","name_en":"A","name_vi":"a","type":"object",'
        '"parent_id":"categoryHaNoi","description_en":"d","description_vi":"d",'
        '"visual_cues_en":"v","visual_cues_vi":"v","tags":[]},'
        '{"id":"cat","name_en":"Cat","name_vi":"cat","type":"category",'
        '"parent_id":null,"description_en":"d","description_vi":"d",'
        '"visual_cues_en":"","visual_cues_vi":"","tags":[]}]'
    )
    nodes = load_kb(kb_file, only_objects=True)
    assert set(nodes.keys()) == {"a"}
    assert nodes["a"].name_en == "A"


@pytest.mark.skipif(
    not Path("data/kb.json").exists(),
    reason="data/kb.json not yet vendored (Task 0.3 pending)",
)
def test_load_kb_real_sample():
    nodes = load_kb(Path("data/kb.json"), only_objects=True)
    assert len(nodes) == 15
    assert "temple_of_literature" in nodes
