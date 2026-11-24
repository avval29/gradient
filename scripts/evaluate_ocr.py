"""Measure handwriting-reading accuracy on your own photos, and what OCR errors do to the grade.

Folder layout: for every photo  name.jpg  (or .png/.jpeg)  put the true text in  name.txt.
Usage:
  python scripts/evaluate_ocr.py --dir data/private/handwriting --out docs/ocr_eval.md
  python scripts/evaluate_ocr.py --dir ... --rubric examples/photosynthesis_rubric.json    # + grade impact

Keep photos of real students' work out of git (data/private/ is ignored) unless you have permission.
Reported: character / word error rate (lower is better), and with a rubric, how often the draft grade
differs between the true text and the OCR text.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image  # noqa: E402

from gradient.evaluate import edit_distance, ocr_error_rates  # noqa: E402
from gradient.grading import grade_answer, load_rubric  # noqa: E402
from gradient.ocr import TrOCRReader, join_lines  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main(argv=None, reader=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rubric", default=None)
    ap.add_argument("--out", default="docs/ocr_eval.md")
    ap.add_argument("--model", default=os.getenv("GRADIENT_OCR_MODEL", "microsoft/trocr-base-handwritten"))
    args = ap.parse_args(argv)

    pairs = [(p, p.with_suffix(".txt")) for p in sorted(Path(args.dir).iterdir())
             if p.suffix.lower() in IMAGE_SUFFIXES and p.with_suffix(".txt").exists()]
    if not pairs:
        raise SystemExit("No image + .txt pairs found.")
    reader = reader or TrOCRReader(args.model)
    truths, preds, doubtful = [], [], 0
    for img_path, txt_path in pairs:
        lines = reader.read(Image.open(img_path))
        preds.append(join_lines(lines))
        truths.append(txt_path.read_text(encoding="utf-8"))
        doubtful += any(l.doubtful for l in lines)

    rates = ocr_error_rates(preds, truths)
    per_image = sorted(((edit_distance(" ".join(p.split()), " ".join(t.split())) / max(1, len(" ".join(t.split()))), path.name)
                        for p, t, (path, _) in zip(preds, truths, pairs)), reverse=True)
    lines = [f"# Handwriting reading: {len(pairs)} photos", "", "| Metric | Value |", "|---|---|",
             f"| Character error rate | {rates['cer']:.1f}% |", f"| Word error rate | {rates['wer']:.1f}% |",
             f"| Photos with at least one low-confidence line | {doubtful}/{len(pairs)} |"]
    if args.rubric:
        rubric = load_rubric(args.rubric)
        diffs = [abs(grade_answer(t, rubric).suggested_score - grade_answer(p, rubric).suggested_score)
                 for p, t in zip(preds, truths)]
        lines += [f"| Mean absolute change in draft score caused by OCR (max {rubric.max_score:g}) | {sum(diffs) / len(diffs):.2f} |",
                  f"| Photos whose draft score changed | {sum(d > 0 for d in diffs)}/{len(diffs)} |"]
    lines += ["", "Worst photos by character error rate: " + ", ".join(f"{n} ({e:.0%})" for e, n in per_image[:3]), "",
              "Photos are the author's own samples; results depend heavily on handwriting and photo quality."]
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
