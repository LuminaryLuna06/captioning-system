# tests/eval/test_eval_summary.py
import pytest
from scripts.eval.eval_summary import aggregate_scores, format_latex_row


def test_aggregate_mean_std():
    scores = [
        {"bleu4": 0.10, "rouge_l": 0.30},
        {"bleu4": 0.20, "rouge_l": 0.40},
        {"bleu4": 0.30, "rouge_l": 0.50},
    ]
    agg = aggregate_scores(scores, ["bleu4", "rouge_l"])
    assert agg["bleu4"]["mean"] == pytest.approx(0.2, abs=1e-4)
    assert agg["rouge_l"]["mean"] == pytest.approx(0.4, abs=1e-4)
    assert agg["bleu4"]["n"] == 3


def test_aggregate_missing_key_skipped():
    scores = [{"bleu4": 0.1}, {"bleu4": 0.2}, {}]
    agg = aggregate_scores(scores, ["bleu4"])
    assert agg["bleu4"]["n"] == 2


def test_aggregate_single_value_std_zero():
    scores = [{"x": 0.5}]
    agg = aggregate_scores(scores, ["x"])
    assert agg["x"]["std"] == pytest.approx(0.0)


def test_format_latex_row():
    row = format_latex_row("BLEU-4", {"mean": 0.1234, "std": 0.0456})
    assert "BLEU-4" in row
    assert "0.1234" in row
    assert "0.0456" in row
    assert r"\pm" in row
    assert row.endswith(r" \\")
