"""Benchmark the Gradient grading engine on the Mohler short-answer dataset.

Usage: python scripts/evaluate_mohler.py --out docs/mohler_eval.md
Needs: pip install datasets   (small download from Hugging Face)

Questions are split into a development half and a test half (disjoint). Calibrations are fitted on the
development half only; every number in the table is measured on the unseen test questions.
Systems (all scored against the average of the two human graders, 0-5):
  - human grader 2 vs grader 1   : agreement between two experts (a noisier target, so not a strict ceiling)
  - always predict dev median    : trivial baseline (marks are skewed high)
  - answer length (calibrated)   : do keywords add anything beyond "longer answers score higher"?
  - Gradient raw                 : draft score from a rubric auto-built from the reference answer
  - Gradient calibrated          : the raw score mapped to human marks (isotonic fit on dev questions)
"""
import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.asag import fit_calibrator, group_rows, score_table, split_by_question  # noqa: E402
from gradient.evaluate import agreement  # noqa: E402


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/mohler_eval.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    groups = group_rows(list(load_dataset("nkazi/MohlerASAG", name="raw", split="open_ended")))
    dev_groups, test_groups = split_by_question(groups, args.dev_fraction, args.seed)
    dev, test = score_table(dev_groups), score_table(test_groups)

    gold = [x.gold for x in test]
    median = statistics.median(x.gold for x in dev)
    grade_cal = fit_calibrator([x.pred for x in dev], [x.gold for x in dev])
    length_cal = fit_calibrator([x.words for x in dev], [x.gold for x in dev])

    systems = [
        ("Human: grader 2 vs grader 1", agreement([x.grader2 for x in test], [x.grader1 for x in test])),
        (f"Baseline: always predict {median:g} (dev median)", agreement([median] * len(test), gold)),
        ("Baseline: answer length (calibrated)", agreement(list(length_cal.predict([x.words for x in test])), gold)),
        ("Gradient: raw keyword-coverage score", agreement([x.pred for x in test], gold)),
        ("Gradient: calibrated on dev questions", agreement(list(grade_cal.predict([x.pred for x in test])), gold)),
    ]
    lines = [f"# Mohler short-answer grading: {len(test)} test answers on {len(test_groups)} unseen questions", "",
             f"Calibrations were fitted on {len(dev)} answers from {len(dev_groups)} other questions (seed {args.seed}). "
             "Scores are 0-5, compared with the average of the two human graders.", "",
             "| System | MAE (lower is better) | Pearson r | Within 1 point | QWK |", "|---|---|---|---|---|"]
    for name, a in systems:
        lines.append(f"| {name} | {fmt(a['mae'])} | {fmt(a['pearson'])} | {a['within1']:.0f}% | {fmt(a['qwk'])} |")
    lines += ["", "Data: Mohler, Bunescu and Mihalcea (2011), via https://huggingface.co/datasets/nkazi/MohlerASAG (CC-BY-4.0).",
              "Rubrics are auto-built from each question's reference answer; teacher-written rubrics may do better.",
              "The human row compares two individual graders, a noisier target than the two-grader average used for the other rows."]
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
