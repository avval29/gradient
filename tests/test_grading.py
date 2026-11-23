import json

import numpy as np
import pytest

from gradient.evaluate import agreement
from gradient.grading import (finalize, grade_answer, load_rubric, log_final, parse_rubric_text)

RUBRIC_TEXT = """
# points | name | description | keywords
2 | Inputs | Uses sunlight, water and carbon dioxide | sunlight|light, water, carbon dioxide|co2
1 | Chlorophyll | Chlorophyll absorbs light | chlorophyll
"""


def rubric():
    return parse_rubric_text("How do plants make food?", RUBRIC_TEXT)


def test_parse_rubric_ok_and_max_score():
    r = rubric()
    assert len(r.criteria) == 2 and r.max_score == 3
    assert r.criteria[0].keywords == ("sunlight|light", "water", "carbon dioxide|co2")


@pytest.mark.parametrize("bad", ["", "# only comment", "x | a | b", "2 | only two", "-1 | a | b"])
def test_parse_rubric_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_rubric_text("q", bad)


def test_full_marks_answer():
    d = grade_answer("Plants use light, water and CO2. Chlorophyll absorbs the light.", rubric())
    assert d.suggested_score == 3 and d.status == "draft"
    assert d.results[0].matched and not d.results[0].missing


def test_partial_and_missing_terms_are_reported():
    d = grade_answer("Plants use sunlight and water.", rubric())
    first = d.results[0]
    assert first.missing == ["carbon dioxide"] and 0 < first.points_awarded < first.points_possible
    assert "To improve - Chlorophyll" in d.feedback


def test_empty_answer_scores_zero_and_is_flagged():
    d = grade_answer("", rubric())
    assert d.suggested_score == 0 and d.needs_review


def test_spelling_mistakes_are_tolerated():
    d = grade_answer("Plants use sunlite, watter and carbon dioxide. Chlorophil absorbs light.", rubric())
    assert d.suggested_score == 3


def test_plural_and_alternatives():
    d = grade_answer("Plants use light, water and co2 and chlorophylls.", rubric())
    assert d.suggested_score == 3


def test_semantic_encoder_can_add_credit_beyond_keywords():
    import zlib

    def enc(texts, dim=128):
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % dim] += 1
        return out

    r = parse_rubric_text("q", "1 | Idea | green pigment absorbs energy | zzz")  # keyword never matches
    answer = "green pigment absorbs energy"
    assert grade_answer(answer, r).suggested_score == 0
    assert grade_answer(answer, r, encoder=enc).suggested_score == 1


def test_finalize_requires_teacher_and_valid_score():
    d = grade_answer("Plants use light, water and CO2.", rubric())
    with pytest.raises(ValueError):
        finalize(d, teacher="  ")
    with pytest.raises(ValueError):
        finalize(d, teacher="Ms Rao", score=99)
    ok = finalize(d, teacher="Ms Rao")
    assert ok.score == d.suggested_score and not ok.overridden
    changed = finalize(d, teacher="Ms Rao", score=3, feedback="Good.")
    assert changed.overridden and changed.feedback == "Good."


def test_log_final_writes_jsonl(tmp_path):
    r = rubric()
    d = grade_answer("Plants use light.", r)
    f = finalize(d, teacher="Ms Rao", score=1)
    path = tmp_path / "log.jsonl"
    log_final(f, d, r, "Plants use light.", path)
    rec = json.loads(path.read_text().strip())
    assert rec["teacher"] == "Ms Rao" and rec["overridden"] is True and rec["suggested_score"] == d.suggested_score


def test_load_rubric_json_example():
    r = load_rubric("examples/photosynthesis_rubric.json")
    assert r.max_score == 5


def test_agreement_metrics():
    a = agreement([5, 4, 3, 0], [5, 4, 3, 0])
    assert a["mae"] == 0 and a["exact"] == 100 and a["qwk"] == pytest.approx(1.0)
    b = agreement([5, 4, 3, 0], [4, 4, 1, 0])
    assert b["mae"] == pytest.approx(0.75) and b["within1"] == 75
    constant = agreement([3, 3], [3, 3])  # kappa is undefined for constant labels: must not crash
    assert constant["mae"] == 0 and constant["exact"] == 100
