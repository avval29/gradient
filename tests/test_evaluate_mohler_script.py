"""Runs scripts/evaluate_mohler.py end to end on synthetic rows, with the dataset download faked."""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_script_runs_offline_and_writes_report(tmp_path, monkeypatch):
    rows = []
    for q in range(12):
        ref = f"concept{q} alpha beta gamma delta epsilon"
        for a, (text, sc) in enumerate([(ref, 5), (f"concept{q} alpha beta", 3.5), (f"concept{q}", 2), ("none", 0)]):
            rows.append({"id": f"E01.Q{q:02d}.A{a:02d}", "question": f"Q{q}", "instructor_answer": ref,
                         "student_answer": text, "score_grader_1": sc, "score_grader_2": sc, "score_avg": sc})
    fake = types.SimpleNamespace(load_dataset=lambda *a, **k: rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    import evaluate_mohler

    out = tmp_path / "report.md"
    evaluate_mohler.main(["--out", str(out)])
    text = out.read_text()
    assert "Gradient: calibrated" in text and "unseen questions" in text and "answer length" in text
