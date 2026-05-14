"""Shared pytest fixtures."""
from pathlib import Path
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that load real models or hit a GPU")


@pytest.fixture(scope="session")
def fixture_video(tmp_path_factory) -> Path:
    """Build a deterministic 12s test video from three existing image fixtures.

    Layout (5 fps, 60 frames total):
        frames  0..19  -> temple_of_literature_1.jpg
        frames 20..39  -> hoangthanh.jpg
        frames 40..59  -> gahanoi.jpg
    """
    import cv2
    import numpy as np
    from PIL import Image

    src_dir = Path(__file__).parent / "fixtures"
    triplet = [
        src_dir / "temple_of_literature_1.jpg",
        src_dir / "hoangthanh.jpg",
        src_dir / "gahanoi.jpg",
    ]
    for p in triplet:
        if not p.exists():
            pytest.skip(f"required fixture missing: {p}")

    out_dir = tmp_path_factory.mktemp("video")
    out_path = out_dir / "synthetic_tour.mp4"

    # Normalise all source images to a common size (smallest common WxH).
    pil_imgs = [Image.open(p).convert("RGB") for p in triplet]
    w = min(im.size[0] for im in pil_imgs)
    h = min(im.size[1] for im in pil_imgs)
    w -= w % 2
    h -= h % 2  # cv2 mp4 writer needs even dimensions
    resized = [im.resize((w, h), Image.LANCZOS) for im in pil_imgs]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 5
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for im in resized:
        bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        for _ in range(fps * 4):  # 4 seconds per image
            writer.write(bgr)
    writer.release()
    assert out_path.exists() and out_path.stat().st_size > 0
    return out_path


@pytest.fixture(scope="session")
def kb_nodes_real():
    """Load the real KB. Skipped if not vendored."""
    kb_path = Path("data/kb.json")
    if not kb_path.exists():
        pytest.skip(f"data/kb.json not present")
    from hanoi_caption.kb_loader import load_kb
    return load_kb(kb_path, only_objects=True)
