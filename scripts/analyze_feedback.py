"""Turn the grading and generation logs into a monitoring report.

Usage: python scripts/analyze_feedback.py --out docs/feedback_report.md
The report contains aggregates only (no student answers), but check it before committing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gradient.feedback import (DEFAULT_GENERATION_LOG, MIN_RECORDS_FOR_CALIBRATION, read_jsonl,  # noqa: E402
                               suggest_calibration, summarize_generation, summarize_grading)

DEFAULT_GRADING_LOG = "data/private/grading_log.jsonl"


def advice(g: dict) -> list[str]:
    if g["n"] < MIN_RECORDS_FOR_CALIBRATION:
        return [f"Only {g['n']} graded answers so far: too few to draw conclusions (aim for {MIN_RECORDS_FOR_CALIBRATION}+)."]
    out = []
    if g["raised"] > 2 * max(1, g["lowered"]):
        out.append("Teachers mostly RAISE the suggested score: Gradient looks too strict. Add keyword alternatives to rubrics "
                   "or consider the calibration below.")
    elif g["lowered"] > 2 * max(1, g["raised"]):
        out.append("Teachers mostly LOWER the suggested score: Gradient may be over-crediting (e.g. keyword stuffing).")
    if g["overrides_flagged_pct"] is not None and g["overrides_flagged_pct"] < 50:
        out.append("Fewer than half of the corrections were on drafts flagged for review: the review flags miss many problem cases.")
    if g["override_rate"] > 50:
        out.append("Teachers change more than half of the drafts: the grader is not yet saving them much effort.")
    return out or ["No clear problem pattern; keep collecting data."]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grading", default=DEFAULT_GRADING_LOG)
    ap.add_argument("--generation", default=DEFAULT_GENERATION_LOG)
    ap.add_argument("--out", default="docs/feedback_report.md")
    args = ap.parse_args(argv)

    grading_records = read_jsonl(args.grading)
    g, gen = summarize_grading(grading_records), summarize_generation(read_jsonl(args.generation))
    lines = ["# Feedback and monitoring report", "", "## Grading corrections", ""]
    if g["n"] == 0:
        lines.append("No grading records yet.")
    else:
        pct = lambda x: "n/a" if x is None else f"{x:.0f}%"  # noqa: E731
        lines += ["| Metric | Value |", "|---|---|",
                  f"| Graded answers | {g['n']} |",
                  f"| Teacher changed the suggested score | {g['overridden']} ({g['override_rate']:.0f}%) |",
                  f"| Mean change (absolute) | {g['mean_abs_change']:.2f} points ({g['mean_abs_change_pct_of_max']:.0f}% of the maximum) |",
                  f"| Raised / lowered | {g['raised']} / {g['lowered']} |",
                  f"| Drafts flagged for review | {g['flagged_rate']:.0f}% |",
                  f"| Corrections that were on flagged drafts | {pct(g['overrides_flagged_pct'])} ({g['unflagged_overrides']} unflagged) |", ""]
        if g["questions"]:
            lines += ["Questions with the most corrections (3+ answers):", ""]
            lines += [f"- {q[:70]}: {rate:.0f}% changed of {n}" for q, n, rate in g["questions"]]
            lines.append("")
        lines += ["**Reading:** " + a for a in advice(g)]
        cal = suggest_calibration(grading_records)
        if cal:
            lines += ["", "Suggested calibration (draft fraction -> teacher fraction), for inspection only:", ""]
            lines += [f"- {x:.2f} -> {y:.2f}" for x, y in cal]
    lines += ["", "## Generated drafts", ""]
    if gen["n"] == 0:
        lines.append("No ratings yet.")
    else:
        lines += [f"- Ratings: {gen['n']}, thumbs-down rate {gen['down_rate']:.0f}%"]
        lines += [f"- {m}: {n} ratings, {rate:.0f}% thumbs-down" for m, (n, rate) in gen["by_mode"].items()]
        if gen["recent_complaints"]:
            lines += ["", "Recent complaints:"] + [f"- {c}" for c in gen["recent_complaints"]]
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
