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


def evaluate_split(groups, seed, dev_fraction):
    """Run every system on one dev/test split by question. Returns (systems, info)."""
    dev_groups, test_groups = split_by_question(groups, dev_fraction, seed)
    dev, test = score_table(dev_groups), score_table(test_groups)
    gold = [x.gold for x in test]
    median = statistics.median(x.gold for x in dev)
    grade_cal = fit_calibrator([x.pred for x in dev], [x.gold for x in dev])
    length_cal = fit_calibrator([x.words for x in dev], [x.gold for x in dev])
    systems = [
        ("Human: grader 2 vs grader 1", agreement([x.grader2 for x in test], [x.grader1 for x in test])),
        ("Baseline: always predict the dev median", agreement([median] * len(test), gold)),
        ("Baseline: answer length (calibrated)", agreement(list(length_cal.predict([x.words for x in test])), gold)),
        ("Gradient: raw keyword-coverage score", agreement([x.pred for x in test], gold)),
        ("Gradient: calibrated on dev questions", agreement(list(grade_cal.predict([x.pred for x in test])), gold)),
    ]
    info = dict(n_dev=len(dev), n_test=len(test), q_dev=len(dev_groups), q_test=len(test_groups), median=median)
    return systems, info


def mean_sd(values, nd=2):
    vals = [v for v in values if v is not None]
    if not vals:
        return "n/a"
    if len(vals) < 2:
        return f"{vals[0]:.{nd}f}"
    return f"{statistics.mean(vals):.{nd}f} ± {statistics.stdev(vals):.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/mohler_eval.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=1, help="repeat with this many different random question splits")
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    args = ap.parse_args(argv)

    from datasets import load_dataset

    groups = group_rows(list(load_dataset("nkazi/MohlerASAG", name="raw", split="open_ended")))
    seeds = [args.seed + i for i in range(args.n_splits)]
    runs = [evaluate_split(groups, sd, args.dev_fraction) for sd in seeds]
    info = runs[0][1]
    header = ["| System | MAE (lower is better) | Pearson r | Within 1 point | QWK |", "|---|---|---|---|---|"]

    if len(runs) == 1:
        systems = runs[0][0]
        lines = [f"# Mohler short-answer grading: {info['n_test']} test answers on {info['q_test']} unseen questions", "",
                 f"Calibrations were fitted on {info['n_dev']} answers from {info['q_dev']} other questions (seed {seeds[0]}). "
                 "Scores are 0-5, compared with the average of the two human graders.", ""] + header
        for name, a in systems:
            name = name.replace("the dev median", f"{info['median']:g} (dev median)")
            lines.append(f"| {name} | {fmt(a['mae'])} | {fmt(a['pearson'])} | {a['within1']:.0f}% | {fmt(a['qwk'])} |")
    else:
        lines = [f"# Mohler short-answer grading: mean ± sd over {len(runs)} random question splits", "",
                 f"Each split fits calibrations on about {info['q_dev']} questions and tests on the other ~{info['q_test']} "
                 f"(seeds {seeds[0]}-{seeds[-1]}). Splits reuse the same 81 questions, so they are not independent: the spread "
                 "shows sensitivity to the split, not uncertainty about new subjects. Scores are 0-5, compared with the average "
                 "of the two human graders.", ""] + header
        for i, (name, _) in enumerate(runs[0][0]):
            col = lambda k: [run[0][i][1][k] for run in runs]  # noqa: E731
            lines.append(f"| {name} | {mean_sd(col('mae'))} | {mean_sd(col('pearson'))} | "
                         f"{mean_sd(col('within1'), 0)}% | {mean_sd(col('qwk'))} |")
    lines += ["", "Data: Mohler, Bunescu and Mihalcea (2011), via https://huggingface.co/datasets/nkazi/MohlerASAG (CC-BY-4.0).",
              "Rubrics are auto-built from each question's reference answer; teacher-written rubrics may do better.",
              "The human row compares two individual graders, a noisier target than the two-grader average used for the other rows.",
              "'Within 1 point' flatters any guess near the top because marks are skewed high; rely on correlation and QWK."]
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
