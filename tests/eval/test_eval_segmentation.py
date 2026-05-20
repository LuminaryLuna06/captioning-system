# tests/eval/test_eval_segmentation.py
import pytest
from scripts.eval.eval_segmentation import tiou, match_segments, seg_metrics, lid_accuracy, kb_node_precision


# --- tiou ---

def test_tiou_perfect_overlap():
    assert tiou(0.0, 10.0, 0.0, 10.0) == pytest.approx(1.0)


def test_tiou_no_overlap():
    assert tiou(0.0, 5.0, 6.0, 10.0) == pytest.approx(0.0)


def test_tiou_partial_overlap():
    # overlap=5, union=15
    assert tiou(0.0, 10.0, 5.0, 15.0) == pytest.approx(5 / 15)


def test_tiou_contained():
    # pred fully inside gt: overlap=5, union=10
    assert tiou(2.0, 7.0, 0.0, 10.0) == pytest.approx(5 / 10)


def test_tiou_adjacent_no_overlap():
    assert tiou(0.0, 5.0, 5.0, 10.0) == pytest.approx(0.0)


# --- match_segments ---

PRED = [
    {"start_s": 0.0,  "end_s": 10.0, "kb_id": "temple"},
    {"start_s": 12.0, "end_s": 22.0, "kb_id": "lake"},
]
GT = [
    {"start_time": 0.0,  "end_time": 9.0,  "kb_id": "temple"},
    {"start_time": 11.0, "end_time": 20.0, "kb_id": "lake"},
]


def test_match_segments_both_match():
    matches = match_segments(PRED, GT, threshold=0.5)
    assert len(matches) == 2


def test_match_segments_no_match_below_threshold():
    # shift pred far right — no overlap with GT
    pred = [{"start_s": 50.0, "end_s": 60.0, "kb_id": "x"}]
    matches = match_segments(pred, GT, threshold=0.5)
    assert len(matches) == 0


def test_match_segments_one_to_one():
    # Both preds overlap GT[0]; pred[0] has higher TIoU (1.0) than pred[1] (8/9)
    pred = [
        {"start_s": 0.0, "end_s": 9.0, "kb_id": "temple"},
        {"start_s": 0.0, "end_s": 8.0, "kb_id": "temple"},
    ]
    gt = [{"start_time": 0.0, "end_time": 9.0, "kb_id": "temple"}]
    matches = match_segments(pred, gt, threshold=0.5)
    assert len(matches) == 1
    assert matches[0][1] == 0  # pred[0] wins (higher TIoU)


# --- seg_metrics ---

def test_seg_metrics_perfect():
    matches = [(0, 0, 1.0), (1, 1, 1.0)]
    m = seg_metrics(matches, n_predicted=2, n_gt=2)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_seg_metrics_partial():
    matches = [(0, 0, 0.8)]  # 1 of 2 GT matched, 1 of 3 pred matched
    m = seg_metrics(matches, n_predicted=3, n_gt=2)
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(1 / 2)


def test_seg_metrics_zero_predicted():
    m = seg_metrics([], n_predicted=0, n_gt=2)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1"] == 0.0


# --- lid_accuracy ---

def test_lid_accuracy_all_correct():
    matches = [(0, 0, 0.9), (1, 1, 0.8)]
    pred = [{"kb_id": "temple"}, {"kb_id": "lake"}]
    gt   = [{"kb_id": "temple"}, {"kb_id": "lake"}]
    assert lid_accuracy(matches, pred, gt) == pytest.approx(1.0)


def test_lid_accuracy_half_correct():
    matches = [(0, 0, 0.9), (1, 1, 0.8)]
    pred = [{"kb_id": "temple"}, {"kb_id": "WRONG"}]
    gt   = [{"kb_id": "temple"}, {"kb_id": "lake"}]
    assert lid_accuracy(matches, pred, gt) == pytest.approx(0.5)


def test_lid_accuracy_no_matches():
    assert lid_accuracy([], [], []) == pytest.approx(0.0)


# --- kb_node_precision ---

def test_kb_node_precision_all_correct():
    matches = [(0, 0, 0.9), (1, 1, 0.8)]
    pred = [{"node_id": "node_001"}, {"node_id": "node_002"}]
    gt   = [{"gt_node_id": "node_001"}, {"gt_node_id": "node_002"}]
    assert kb_node_precision(matches, pred, gt) == pytest.approx(1.0)


def test_kb_node_precision_half_correct():
    matches = [(0, 0, 0.9), (1, 1, 0.8)]
    pred = [{"node_id": "node_001"}, {"node_id": "WRONG"}]
    gt   = [{"gt_node_id": "node_001"}, {"gt_node_id": "node_002"}]
    assert kb_node_precision(matches, pred, gt) == pytest.approx(0.5)


def test_kb_node_precision_no_matches():
    assert kb_node_precision([], [], []) == pytest.approx(0.0)


# --- run_evaluation ---

def test_run_evaluation_empty_in_kb():
    """Empty in_kb must not crash (statistics.mean on empty sequence)."""
    from scripts.eval.eval_segmentation import run_evaluation
    test_set = {"in_kb": [], "out_of_kb": []}
    result = run_evaluation(test_set, [])
    assert result["n_in_kb_videos"] == 0
    assert result["thresholds"]["0.5"]["f1"] == 0.0


def test_run_evaluation_basic():
    from scripts.eval.eval_segmentation import run_evaluation
    test_set = {
        "in_kb": [{
            "video_id": "v1",
            "filename": "A.MOV",
            "gt_segments": [{"start_time": 0.0, "end_time": 10.0, "kb_id": "temple", "gt_node_id": "n1"}],
        }],
        "out_of_kb": [{"video_id": "v2", "filename": "B.MOV", "duration": 5.0}],
    }
    results = [{
        "video_id": "v1",
        "predicted_segments": [{"start_s": 0.0, "end_s": 10.0, "kb_id": "temple", "node_id": "n1"}],
    }]
    out = run_evaluation(test_set, results)
    assert out["thresholds"]["0.5"]["recall"] == pytest.approx(1.0)
    assert out["thresholds"]["0.5"]["lid_acc"] == pytest.approx(1.0)
    assert out["refusal_rate"] == pytest.approx(1.0)  # v2 not in results → counted as refusal
