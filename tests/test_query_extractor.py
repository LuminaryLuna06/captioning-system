from hanoi_caption.query_extractor import parse_queries


def test_parse_queries_strips_markdown_fences():
    raw = '```json\n["a", "b", "c"]\n```'
    assert parse_queries(raw) == ["a", "b", "c"]


def test_parse_queries_handles_extra_prose():
    raw = 'Here are the queries: ["red gate", "stone stele"]. Done.'
    assert parse_queries(raw) == ["red gate", "stone stele"]


def test_parse_queries_dedupes_and_strips():
    raw = '["  Stele ", "stele", "Stele"]'
    out = parse_queries(raw)
    assert out == ["stele"]


def test_parse_queries_returns_empty_on_garbage():
    assert parse_queries("not json at all") == []
