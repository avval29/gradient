import json
import sys
from pathlib import Path

import pytest

from gradient.feedback import (log_generation_feedback, read_jsonl, suggest_calibration, summarize_generation,
                               summarize_grading)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def rec(suggested, final, flagged, q="Q1", max_score=5):
    return {"question": q, "score": final, "suggested_score": suggested, "max_score": max_score,
            "overridden": final != suggested, "needs_review": flagged}


RECORDS = [rec(3, 4, True), rec(5, 5, False), rec(2, 1, True), rec(4, 3, False)]


def test_grading_summary_arithmetic():
    s = summarize_grading(RECORDS)
    assert s["n"] == 4 and s["overridden"] == 3 and s["override_rate"] == pytest.approx(75)
    assert s["mean_abs_change"] == pytest.approx(0.75) and s["mean_signed_change"] == pytest.approx(-0.25)
    assert (s["raised"], s["lowered"]) == (1, 2)
    assert s["flagged_rate"] == pytest.approx(50)
    assert s["overrides_flagged_pct"] == pytest.approx(200 / 3) and s["unflagged_overrides"] == 1


def test_grading_summary_empty_and_malformed_records_are_ignored():
    assert summarize_grading([]) == {"n": 0}
    assert summarize_grading([{"foo": 1}, {"score": 1}])["n"] == 0
    assert summarize_grading([rec(1, 1, False, max_score=0)])["n"] == 0


def test_questions_with_most_corrections_need_three_answers():
    rows = [rec(1, 2, False, q="A")] * 3 + [rec(1, 1, False, q="B")] * 3 + [rec(1, 2, False, q="C")]
    qs = summarize_grading(rows)["questions"]
    assert [q[0] for q in qs] == ["A", "B"] and qs[0][2] == 100


def test_read_jsonl_skips_bad_lines_and_missing_file(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n[1, 2]\n{"b": 2}\n')
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_generation_feedback_roundtrip_and_summary(tmp_path):
    p = tmp_path / "gen.jsonl"
    log_generation_feedback("quiz (Q&A)", "photosynthesis", "up", path=p)
    log_generation_feedback("quiz (Q&A)", "photosynthesis", "down", "too easy", "draft", "src", p)
    log_generation_feedback("worksheet", "cells", "down", "", "draft", "src", p)
    with pytest.raises(ValueError):
        log_generation_feedback("quiz (Q&A)", "t", "meh", path=p)
    s = summarize_generation(read_jsonl(p))
    assert s["n"] == 3 and s["down_rate"] == pytest.approx(200 / 3)
    assert s["by_mode"]["worksheet"] == (1, 100.0) and s["recent_complaints"] == ["too easy"]
    assert summarize_generation([]) == {"n": 0}


def test_calibration_needs_enough_varied_data_and_is_monotone():
    assert suggest_calibration(RECORDS) is None  # too few records
    rows = [rec(s, min(5, s + 1), False) for s in [0, 1, 2, 3, 4, 5] * 6]  # teachers always give +1
    knots = suggest_calibration(rows)
    assert knots and [y for _, y in knots] == sorted(y for _, y in knots)
    assert knots[0][1] > knots[0][0]  # teachers are more generous than the draft at the low end
    assert suggest_calibration([rec(3, 4, False)] * 40) is None  # no variety in draft scores


def test_app_thumbs_log_feedback(tmp_path, monkeypatch):
    import app

    path = tmp_path / "fb.jsonl"
    monkeypatch.setattr(app, "FEEDBACK_LOG", str(path))
    assert app.thumbs_down("quiz (Q&A)", "photosynthesis", "1. Q?", "a.txt#0", "too easy") == "Thanks, your feedback was saved."
    assert app.thumbs_up("quiz (Q&A)", "t", "   ", "", "") == "Generate a draft first."
    saved = json.loads(path.read_text().strip())
    assert saved["rating"] == "down" and saved["comment"] == "too easy"


def test_analysis_script_writes_report(tmp_path):
    import analyze_feedback

    grading = tmp_path / "grading.jsonl"
    grading.write_text("\n".join(json.dumps(r) for r in RECORDS))
    out = tmp_path / "report.md"
    analyze_feedback.main(["--grading", str(grading), "--generation", str(tmp_path / "none.jsonl"), "--out", str(out)])
    text = out.read_text()
    assert "Graded answers | 4" in text and "too few to draw conclusions" in text and "No ratings yet." in text
    analyze_feedback.main(["--grading", str(tmp_path / "x.jsonl"), "--generation", str(tmp_path / "y.jsonl"), "--out", str(out)])
    assert "No grading records yet." in out.read_text()
