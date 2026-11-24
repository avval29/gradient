import pytest

from gradient.asag import _to_five_point, clean, group_rows, predict_scores, rubric_from_reference


def row(id_, ans, avg, g1=None, g2=None, q="What does a stack do?", ref="A stack stores items last in first out."):
    return {"id": id_, "question": q, "instructor_answer": ref, "student_answer": ans,
            "score_grader_1": avg if g1 is None else g1, "score_grader_2": avg if g2 is None else g2,
            "score_avg": avg}


def test_rubric_from_reference_equal_shares_and_dedup():
    r = rubric_from_reference("q", "A stack stores items; stacks store items last in first out.", max_score=5)
    names = [c.name for c in r.criteria]
    assert names.count("stack") + names.count("stacks") == 1  # plural deduplicated
    assert r.max_score == pytest.approx(5)
    assert len({round(c.points, 6) for c in r.criteria}) == 1


def test_rubric_none_when_reference_has_no_content_words():
    assert rubric_from_reference("q", "It is the.") is None


def test_group_rows_by_question_prefix():
    rows = [row("E01.Q01.A00", "a", 5), row("E01.Q01.A01", "b", 3),
            row("E01.Q02.A00", "c", 4, q="Other?", ref="Something else.")]
    groups = group_rows(rows)
    assert [g.key for g in groups] == ["E01.Q01", "E01.Q02"]
    assert groups[0].answers == ["a", "b"] and groups[0].scores == [5.0, 3.0]


def test_ten_point_grader_scores_are_halved():
    assert _to_five_point(8, 10, 4.5) == (4, 5)      # raw 0-10 columns, average already on 0-5
    assert _to_five_point(4, 5, 4.5) == (4, 5)       # already 0-5


def test_better_answers_get_higher_predictions():
    ref = "A stack stores items in last in first out order."
    rows = [row("E01.Q01.A00", "A stack stores items in last in first out order.", 5, ref=ref),
            row("E01.Q01.A01", "It stores items.", 2.5, ref=ref),
            row("E01.Q01.A02", "I like pizza.", 0, ref=ref)]
    pred, gold, g1, g2 = predict_scores(group_rows(rows))
    assert pred[0] > pred[1] > pred[2] and pred[0] == pytest.approx(5) and gold == [5, 2.5, 0]


def test_clean_strips_html_but_keeps_code_comparisons():
    assert clean("a<br><br>b &amp; c<br/>") == "a b & c"
    assert clean("loop while i<n and j>m") == "loop while i<n and j>m"


def test_loaded_rows_are_cleaned_so_markup_is_not_a_keyword():
    r = row("E01.Q01.A00", "ok<br>", 5, q="Q?<br>", ref="A stack<br><br>stores items &amp; more")
    g = group_rows([r])[0]
    assert g.reference == "A stack stores items & more" and g.answers == ["ok"]
    names = [c.name for c in rubric_from_reference("q", g.reference).criteria]
    assert "br" not in names and "amp" not in names


def test_agreement_rounds_halves_up_not_to_even():
    from gradient.evaluate import agreement

    assert agreement([2.5, 3.5, 4.5], [3, 4, 5])["exact"] == 100


def _many_groups(n=10):
    rows = []
    for q in range(n):
        ref = f"term{q} alpha beta gamma delta"
        for a, (text, sc) in enumerate([(f"term{q} alpha beta gamma delta", 5), (f"term{q} alpha", 3), ("nothing", 0)]):
            rows.append(row(f"E01.Q{q:02d}.A{a:02d}", text, sc, q=f"Q{q}", ref=ref))
    return group_rows(rows)


def test_split_by_question_is_disjoint_complete_and_seeded():
    from gradient.asag import split_by_question

    groups = _many_groups()
    dev, test = split_by_question(groups, 0.5, 42)
    assert {g.key for g in dev}.isdisjoint({g.key for g in test})
    assert len(dev) + len(test) == len(groups) and len(dev) == 5
    assert [g.key for g in split_by_question(groups, 0.5, 42)[0]] == [g.key for g in dev]


def test_score_table_aligns_fields():
    from gradient.asag import score_table

    t = score_table(_many_groups(2))
    assert len(t) == 6 and t[0].gold == 5 and t[0].words == 5 and t[0].pred > t[1].pred > t[2].pred


def test_calibrator_is_monotone_and_fixes_scale():
    from gradient.asag import fit_calibrator

    cal = fit_calibrator([0, 1, 2, 3, 4, 5], [3, 3.5, 4, 4.5, 4.5, 5])  # humans are more generous than raw scores
    out = list(cal.predict([0, 2.5, 5, 99]))
    assert out == sorted(out) and out[-1] == 5  # monotone, clipped beyond the training range


def _bow_encoder(texts, dim=256):
    import zlib

    import numpy as np

    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % dim] += 1
    return out


def test_score_table_adds_semantic_similarity_when_encoder_given():
    from gradient.asag import score_table

    t = score_table(_many_groups(2), encoder=_bow_encoder)
    assert t[0].cos == pytest.approx(1.0) and t[0].cos > t[1].cos > t[2].cos
    assert score_table(_many_groups(2))[0].cos is None
