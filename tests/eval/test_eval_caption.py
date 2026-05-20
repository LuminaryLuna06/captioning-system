# tests/eval/test_eval_caption.py
import pytest
from scripts.eval.eval_caption import caption_metrics_one, kb_coverage


def test_identical_strings_high_scores():
    text = "The ancient Pen Tower stands tall on the eastern bank of Hoan Kiem Lake."
    m = caption_metrics_one(text, text)
    assert m["bleu4"] > 0.9
    assert m["rouge_l"] == pytest.approx(1.0)
    assert m["meteor"] > 0.9


def test_completely_different_strings_low_scores():
    pred = "The weather is sunny today and the birds are singing."
    ref  = "The ancient tower was built in 1865 by scholar Nguyen Van Sieu."
    m = caption_metrics_one(pred, ref)
    assert m["bleu4"] < 0.1
    assert m["rouge_l"] < 0.2
    assert m["meteor"] < 0.2


def test_partial_overlap_mid_scores():
    pred = "The Pen Tower was built in 1865 by a famous scholar in Hanoi."
    ref  = "The ancient Pen Tower stands on Hoan Kiem Lake, built in 1865 by Nguyen Van Sieu."
    m = caption_metrics_one(pred, ref)
    assert 0.1 < m["rouge_l"] < 1.0
    assert 0.1 < m["meteor"] < 1.0


def test_keys_present():
    m = caption_metrics_one("hello world", "hello world")
    assert set(m.keys()) == {"bleu4", "rouge_l", "meteor"}


# --- kb_coverage ---

def test_kb_coverage_full():
    kb = "five-story stone tower built in 1865 by scholar Nguyen Van Sieu"
    cap = "This five-story stone tower was built in 1865 by the scholar Nguyen Van Sieu"
    cov = kb_coverage(kb, cap)
    assert cov > 0.7


def test_kb_coverage_empty_caption():
    assert kb_coverage("five story tower", "") == pytest.approx(0.0)


def test_kb_coverage_no_overlap():
    kb = "lotus pond surrounded by willow trees"
    cap = "The mountain rises above the clouds with snow at the peak"
    assert kb_coverage(kb, cap) < 0.2


def test_kb_coverage_empty_kb():
    assert kb_coverage("", "anything") == pytest.approx(0.0)
