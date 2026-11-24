"""Feedback loop and monitoring (course Module 4).

Two kinds of feedback are logged locally (data/private/ is gitignored; logs hold teacher/student content):
- grading corrections: written by grading.log_final() every time a teacher approves a grade
- generation ratings: thumbs up/down on generated drafts, with an optional comment

summarize_* turn those logs into numbers a person can act on. Nothing here changes Gradient's behaviour
automatically: suggestions are shown to a human, who decides (teacher in the loop applies to the developer too).
"""
import json
import statistics
import time
from pathlib import Path

DEFAULT_GENERATION_LOG = "data/private/generation_feedback.jsonl"
MIN_RECORDS_FOR_CALIBRATION = 30


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON-lines log, skipping blank or corrupted lines. Missing file -> empty list."""
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def log_generation_feedback(mode: str, topic: str, rating: str, comment: str = "", output: str = "",
                            sources: str = "", path: str | Path = DEFAULT_GENERATION_LOG) -> None:
    if rating not in ("up", "down"):
        raise ValueError("rating must be 'up' or 'down'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mode": mode, "topic": topic, "rating": rating,
              "comment": comment.strip(), "sources": sources, "output": output}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


_GRADING_KEYS = ("score", "suggested_score", "max_score", "overridden")


def summarize_grading(records: list[dict]) -> dict:
    """How often and how far teachers change Gradient's draft scores. Scores are on the rubric's own scale."""
    rows = [r for r in records if all(k in r for k in _GRADING_KEYS) and r["max_score"] > 0]
    if not rows:
        return {"n": 0}
    changes = [r["score"] - r["suggested_score"] for r in rows]
    over = [r for r in rows if r["overridden"]]
    flagged_over = [r for r in over if r.get("needs_review")]
    by_q: dict[str, list[bool]] = {}
    for r in rows:
        by_q.setdefault(r.get("question", "?"), []).append(bool(r["overridden"]))
    questions = sorted(((q, len(v), 100 * sum(v) / len(v)) for q, v in by_q.items() if len(v) >= 3),
                       key=lambda t: (-t[2], -t[1]))
    return {
        "n": len(rows),
        "overridden": len(over),
        "override_rate": 100 * len(over) / len(rows),
        "mean_abs_change": statistics.mean(abs(c) for c in changes),
        "mean_abs_change_pct_of_max": 100 * statistics.mean(abs(r["score"] - r["suggested_score"]) / r["max_score"] for r in rows),
        "mean_signed_change": statistics.mean(changes),  # > 0: teachers give more than Gradient suggested
        "raised": sum(c > 0 for c in changes),
        "lowered": sum(c < 0 for c in changes),
        "flagged_rate": 100 * sum(bool(r.get("needs_review")) for r in rows) / len(rows),
        "overrides_flagged_pct": 100 * len(flagged_over) / len(over) if over else None,
        "unflagged_overrides": len(over) - len(flagged_over),
        "questions": questions[:5],
    }


def summarize_generation(records: list[dict]) -> dict:
    rows = [r for r in records if r.get("rating") in ("up", "down")]
    if not rows:
        return {"n": 0}
    by_mode: dict[str, list[bool]] = {}
    for r in rows:
        by_mode.setdefault(r.get("mode", "?"), []).append(r["rating"] == "down")
    return {
        "n": len(rows),
        "down_rate": 100 * sum(r["rating"] == "down" for r in rows) / len(rows),
        "by_mode": {m: (len(v), 100 * sum(v) / len(v)) for m, v in by_mode.items()},
        "recent_complaints": [r["comment"] for r in rows if r["rating"] == "down" and r.get("comment")][-5:],
    }


def suggest_calibration(records: list[dict], min_records: int = MIN_RECORDS_FOR_CALIBRATION):
    """Monotone map from Gradient's draft score to what teachers finally gave, both as fractions of the maximum.
    Returns a list of (draft_fraction, teacher_fraction) knots, or None if there is too little varied data.
    A SUGGESTION for the developer to inspect; it is never applied automatically."""
    rows = [r for r in records if all(k in r for k in _GRADING_KEYS) and r["max_score"] > 0]
    if len(rows) < min_records:
        return None
    xs = [r["suggested_score"] / r["max_score"] for r in rows]
    ys = [r["score"] / r["max_score"] for r in rows]
    if len(set(round(x, 3) for x in xs)) < 3:
        return None
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0, y_max=1, increasing=True, out_of_bounds="clip").fit(xs, ys)
    return [(round(float(x), 3), round(float(y), 3)) for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)]
