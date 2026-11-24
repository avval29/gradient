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


def test_multi_split_report_has_mean_and_sd(tmp_path, monkeypatch):
    rows = []
    for q in range(16):
        ref = f"concept{q} alpha beta gamma delta epsilon"
        for a, (text, sc) in enumerate([(ref, 5), (f"concept{q} alpha beta", 3.5), (f"concept{q}", 2), ("none", 0)]):
            rows.append({"id": f"E01.Q{q:02d}.A{a:02d}", "question": f"Q{q}", "instructor_answer": ref,
                         "student_answer": text, "score_grader_1": sc, "score_grader_2": sc, "score_avg": sc})
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    import evaluate_mohler

    out = tmp_path / "multi.md"
    evaluate_mohler.main(["--out", str(out), "--n-splits", "4"])
    text = out.read_text()
    assert "mean ± sd over 4 random question splits" in text and "±" in text.split("| Gradient: calibrated")[1]


def test_mean_sd_handles_missing_and_single_values():
    import evaluate_mohler as em

    assert em.mean_sd([None, None]) == "n/a" and em.mean_sd([0.5]) == "0.50"
    assert em.mean_sd([0.4, 0.6]) == "0.50 ± 0.14"


def test_semantic_option_adds_rows(tmp_path, monkeypatch):
    import zlib

    import numpy as np

    def enc(texts, dim=256):
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % dim] += 1
        return out

    rows = []
    for q in range(14):
        ref = f"concept{q} alpha beta gamma delta epsilon"
        for a, (text, sc) in enumerate([(ref, 5), (f"concept{q} alpha beta", 3.5), (f"concept{q}", 2), ("none", 0)]):
            rows.append({"id": f"E01.Q{q:02d}.A{a:02d}", "question": f"Q{q}", "instructor_answer": ref,
                         "student_answer": text, "score_grader_1": sc, "score_grader_2": sc, "score_avg": sc})
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    import evaluate_mohler

    monkeypatch.setattr(evaluate_mohler, "sentence_transformer_encoder", lambda: enc)
    out = tmp_path / "sem.md"
    evaluate_mohler.main(["--out", str(out), "--semantic", "--n-splits", "3"])
    text = out.read_text()
    assert "Semantic similarity to reference" in text and "keyword + semantic combined" in text
