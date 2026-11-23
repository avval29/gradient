"""Compare draft grades with a teacher's grades.

Usage:
  python scripts/evaluate_grading.py --rubric examples/photosynthesis_rubric.json \
      --csv examples/photosynthesis_answers.csv
CSV columns: answer,score   (score = the teacher's mark on the same scale as the rubric)

IMPORTANT: examples/ holds hand-written demo data, not a benchmark. For numbers you can quote,
use real student answers graded by a teacher who did not see the draft scores.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.config import make_encoder  # noqa: E402
from gradient.evaluate import agreement  # noqa: E402
from gradient.grading import grade_answer, load_rubric  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None, help="optional markdown report path")
    args = ap.parse_args()

    rubric = load_rubric(args.rubric)
    encoder = make_encoder()
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [(r["answer"], float(r["score"])) for r in csv.DictReader(f)]
    drafts = [grade_answer(a, rubric, encoder) for a, _ in rows]
    pred = [d.suggested_score for d in drafts]
    gold = [s for _, s in rows]

    a = agreement(pred, gold)
    flagged = [d.needs_review for d in drafts]
    errors = [abs(p - g) >= 1 for p, g in zip(pred, gold)]
    caught = sum(f and e for f, e in zip(flagged, errors))
    lines = [f"# Grader vs teacher ({len(rows)} answers, max score {rubric.max_score:g}, "
             f"{'semantic+lexical' if encoder else 'lexical only'})", "",
             "| Metric | Value |", "|---|---|",
             f"| Mean absolute error | {a['mae']:.2f} |",
             f"| Exact agreement (rounded) | {a['exact']:.0f}% |",
             f"| Within 1 point | {a['within1']:.0f}% |",
             f"| Quadratic weighted kappa | {'n/a' if a['qwk'] is None else format(a['qwk'], '.2f')} |",
             f"| Flagged for teacher review | {100 * sum(flagged) / len(rows):.0f}% |",
             f"| Errors of 1+ point that were flagged | {caught}/{sum(errors)} |", "",
             "## Largest disagreements", ""]
    for i in sorted(range(len(rows)), key=lambda i: -abs(pred[i] - gold[i]))[:5]:
        if pred[i] != gold[i]:
            lines.append(f"- draft {pred[i]:g} vs teacher {gold[i]:g} (flagged: {flagged[i]}): {rows[i][0][:100]}")
    print("\n".join(lines))
    if args.out:
        Path(args.out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
