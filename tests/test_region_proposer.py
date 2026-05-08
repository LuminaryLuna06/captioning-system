import numpy as np

from hanoi_caption.region_proposer import filter_regions


class _Detection:
    def __init__(self, box, score, query):
        self.box = box
        self.score = score
        self.query = query


def _det(x1, y1, x2, y2, score=0.9, query="x"):
    return _Detection(box=(x1, y1, x2, y2), score=score, query=query)


def test_filter_drops_tiny_masks():
    image_area = 100 * 100
    dets = [_det(0, 0, 5, 5)]  # 25 px = 0.25% of image
    out = filter_regions(dets, image_area=image_area, min_area_frac=0.01, iou_threshold=0.7, max_keep=6)
    assert out == []


def test_filter_drops_high_iou_overlaps_keeping_higher_score():
    image_area = 100 * 100
    dets = [
        _det(10, 10, 50, 50, score=0.95),
        _det(11, 11, 51, 51, score=0.85),  # nearly identical -> dropped
        _det(60, 60, 90, 90, score=0.8),
    ]
    out = filter_regions(dets, image_area=image_area, min_area_frac=0.01, iou_threshold=0.7, max_keep=6)
    assert len(out) == 2
    assert any(d.score == 0.95 for d in out)
    assert all(d.score != 0.85 for d in out)


def test_filter_caps_max_keep():
    image_area = 100 * 100
    dets = [_det(0, 0, 30, 30 + i, score=0.9 - 0.01 * i) for i in range(10)]
    out = filter_regions(dets, image_area=image_area, min_area_frac=0.001, iou_threshold=0.99, max_keep=4)
    assert len(out) == 4
