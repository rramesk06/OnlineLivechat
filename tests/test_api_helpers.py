from backend.main import heuristic_relevance, normalize_text


def test_normalize_text():
    assert normalize_text(" hello\n  world ") == "hello world"


def test_heuristic_relevance():
    results = [{"text": "The system uses React and Node.js.", "hybrid_score": 0.7}]
    assert heuristic_relevance("What uses React?", results)


def test_heuristic_irrelevance():
    results = [{"text": "The system uses React.", "hybrid_score": 0.02}]
    assert not heuristic_relevance("What is the capital of France?", results)
