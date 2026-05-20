# tests/eval/test_eval_llm_judge.py
import json
import pytest
from unittest.mock import MagicMock, patch
from scripts.eval.eval_llm_judge import build_prompt, parse_response, judge_caption


SAMPLE_ITEM = {
    "name_en": "Pen Tower",
    "kb_description": "Five-story stone tower built in 1865 by scholar Nguyen Van Sieu.",
    "predicted_caption": "The Pen Tower is a historic five-story tower.",
    "reference_caption": "The ancient Pen Tower stands tall on the eastern shore of Hoan Kiem Lake.",
}


def test_build_prompt_contains_key_fields():
    prompt = build_prompt(SAMPLE_ITEM)
    assert "Pen Tower" in prompt
    assert "1865" in prompt
    assert SAMPLE_ITEM["predicted_caption"] in prompt
    assert SAMPLE_ITEM["reference_caption"] in prompt


def test_parse_response_valid():
    raw = '{"factual_accuracy": 4, "visual_grounding": 3, "tone": 5, "hallucination": 4, "reasoning": "good"}'
    scores = parse_response(raw)
    assert scores["factual_accuracy"] == 4
    assert scores["visual_grounding"] == 3
    assert scores["tone"] == 5
    assert scores["hallucination"] == 4
    assert "reasoning" in scores


def test_parse_response_clamps_to_1_5():
    raw = '{"factual_accuracy": 0, "visual_grounding": 6, "tone": 3, "hallucination": 3, "reasoning": "x"}'
    scores = parse_response(raw)
    assert scores["factual_accuracy"] == 1
    assert scores["visual_grounding"] == 5


def test_parse_response_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_response("not json at all")


def test_judge_caption_calls_openai():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps({
        "factual_accuracy": 4, "visual_grounding": 3,
        "tone": 5, "hallucination": 4, "reasoning": "looks good",
    })
    mock_client.chat.completions.create.return_value = mock_response

    result = judge_caption(mock_client, SAMPLE_ITEM)

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    assert call_kwargs["temperature"] == 0.0
    assert result["factual_accuracy"] == 4
