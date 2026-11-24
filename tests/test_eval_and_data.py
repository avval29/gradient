import pytest

from gradient.data import build_examples
from gradient.evaluate import score, score_multi


def test_identical_text_scores_perfect():
    s = score(["what is photosynthesis"], ["what is photosynthesis"])
    assert s["rougeL"] == pytest.approx(100) and s["bleu"] == pytest.approx(100)


def test_different_text_scores_lower():
    s = score(["completely unrelated words here"], ["what is photosynthesis"])
    assert s["rougeL"] < 10


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score(["a"], ["a", "b"])


def test_squad_rows_become_two_tasks():
    row = {"context": "Plants make food.", "question": "What do plants make?",
           "answers": {"text": ["food"], "answer_start": [12]}}
    ex = build_examples([row])
    assert len(ex) == 2
    assert ex[0]["target"] == "What do plants make?" and ex[1]["target"] == "food"


def test_multi_reference_takes_best_match():
    refs = [["who wrote it", "what is photosynthesis"]]
    assert score_multi(["what is photosynthesis"], refs)["rougeL"] == pytest.approx(100)
    assert score_multi(["what is photosynthesis"], refs)["bleu"] == pytest.approx(100)


def test_multi_reference_uneven_counts_ok():
    s = score_multi(["a b c d", "e f g h"], [["a b c d"], ["x y", "e f g h", "z"]])
    assert s["rougeL"] == pytest.approx(100)
