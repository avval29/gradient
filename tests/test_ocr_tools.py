"""Wiring tests for the OCR app tab and scripts, using a fake reader (no model download)."""
import sys
from pathlib import Path

from PIL import Image

from gradient.ocr import OCRLine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class FakeReader:
    def read(self, image):
        return [OCRLine("Plants use light water and CO2", 0.9, 0, 10), OCRLine("chlorofil absorbs lite", 0.3, 10, 20)]


def test_app_read_photo_fills_text_and_flags_doubtful_lines(monkeypatch):
    import app

    monkeypatch.setattr(app, "get_reader", lambda: FakeReader())
    text, note = app.read_photo(Image.new("RGB", (50, 50), "white"))
    assert text.splitlines()[0] == "Plants use light water and CO2"
    assert "correct the text" in note and "line(s): 2" in note
    assert app.read_photo(None)[1] == "Upload a photo first."


def test_evaluate_ocr_reports_error_rates_and_grade_impact(tmp_path):
    import evaluate_ocr

    Image.new("RGB", (60, 40), "white").save(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("Plants use light water and CO2\nchlorophyll absorbs light")
    (tmp_path / "orphan.png").write_bytes(b"x")  # no .txt: must be skipped
    out = tmp_path / "report.md"
    evaluate_ocr.main(["--dir", str(tmp_path), "--out", str(out), "--rubric", "examples/photosynthesis_rubric.json"],
                      reader=FakeReader())
    text = out.read_text()
    assert "1 photos" in text and "Character error rate" in text and "Mean absolute change in draft score" in text
    assert "1/1" in text  # the photo has a low-confidence line


def test_read_handwriting_segment_only_saves_lines(tmp_path, capsys):
    import read_handwriting
    from tests.helpers import fake_page

    fake_page([100, 250]).save(tmp_path / "page.png")
    read_handwriting.main([str(tmp_path / "page.png"), "--segment-only", "--save-lines", str(tmp_path / "lines")])
    assert "2 line(s) found" in capsys.readouterr().out
    assert len(list((tmp_path / "lines").glob("*.png"))) == 2
