"""Rubric-based DRAFT grading of short typed answers.

Design rules
1. Gradient only drafts. A grade becomes final only through finalize(), which needs a named teacher.
2. Every criterion shows its evidence (matched / missing terms, best sentence) so a teacher can check it.
3. Uncertain cases are flagged for review instead of being decided quietly.
4. Teacher corrections are logged (data for improving the grader); logs contain student work, keep them private.

How a criterion is scored
- lexical coverage: share of the criterion's keywords found in the answer. Keywords may list
  alternatives ("sun|sunlight"); matching ignores plurals and tolerates small spelling mistakes.
- optional semantic similarity (needs an `encoder`, e.g. sentence-transformers) between the criterion
  description and the answer's sentences, which helps with paraphrase.
- evidence = max(lexical, semantic). evidence >= FULL -> full points; >= PARTIAL -> half; else zero.

Known limits: lexical matching under-credits correct answers phrased in unexpected words (including
regional-language or non-native phrasing), and it can over-credit keyword stuffing. Hence the review flags.
"""
import json
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from gradient.grounding import content_tokens

FULL, PARTIAL = 0.7, 0.4   # evidence thresholds
BORDER = 0.08              # evidence this close to a threshold is flagged for review
SEM_LOW, SEM_HIGH = 0.25, 0.65  # cosine range mapped to 0..1 evidence
MIN_ANSWER_TOKENS = 3


@dataclass(frozen=True)
class Criterion:
    name: str
    description: str
    points: float
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Rubric:
    question: str
    criteria: tuple[Criterion, ...]

    @property
    def max_score(self) -> float:
        return sum(c.points for c in self.criteria)


@dataclass
class CriterionResult:
    name: str
    points_possible: float
    points_awarded: float
    evidence: float
    matched: list[str]
    missing: list[str]
    best_sentence: str | None


@dataclass
class DraftGrade:
    suggested_score: float
    max_score: float
    results: list[CriterionResult]
    feedback: str
    needs_review: bool
    review_reasons: list[str]
    status: str = "draft"  # never "final": see finalize()


@dataclass(frozen=True)
class FinalGrade:
    score: float
    max_score: float
    feedback: str
    teacher: str
    suggested_score: float
    overridden: bool


# ---------- rubric input ----------

def parse_rubric_text(question: str, text: str) -> Rubric:
    """One criterion per line:  points | name | description | keyword, keyword|alternative, ...

    The keywords column is optional. Lines starting with # are ignored.
    """
    criteria = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 3)]
        if len(parts) < 3:
            raise ValueError(f"line {n}: expected 'points | name | description | keywords'")
        try:
            points = float(parts[0])
        except ValueError:
            raise ValueError(f"line {n}: points must be a number, got '{parts[0]}'") from None
        if points <= 0:
            raise ValueError(f"line {n}: points must be positive")
        keywords = tuple(k.strip() for k in parts[3].split(",") if k.strip()) if len(parts) == 4 else ()
        criteria.append(Criterion(parts[1], parts[2], points, keywords))
    if not criteria:
        raise ValueError("rubric has no criteria")
    return Rubric(question, tuple(criteria))


def load_rubric(path: str | Path) -> Rubric:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return Rubric(d["question"], tuple(
        Criterion(c["name"], c["description"], float(c["points"]), tuple(c.get("keywords", [])))
        for c in d["criteria"]))


# ---------- matching ----------

def _token_present(token: str, answer_tokens: set[str]) -> bool:
    if token in answer_tokens:
        return True
    if len(token) < 5:
        return False
    return any(len(t) >= 5 and SequenceMatcher(None, token, t).ratio() >= 0.85 for t in answer_tokens)


def _keyword_hit(keyword: str, answer_tokens: set[str]) -> bool:
    """A keyword is hit if ANY of its '|' alternatives has ALL its content words in the answer."""
    for alt in keyword.split("|"):
        toks = content_tokens(alt)
        if toks and all(_token_present(t, answer_tokens) for t in toks):
            return True
    return False


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]


def _semantic(description: str, answer: str, encoder) -> float:
    sents = _sentences(answer) or [answer]
    vecs = np.asarray(encoder([description] + sents), dtype=float)
    vecs = vecs / np.where((n := np.linalg.norm(vecs, axis=1, keepdims=True)) == 0, 1, n)
    best = float((vecs[1:] @ vecs[0]).max())
    return min(1.0, max(0.0, (best - SEM_LOW) / (SEM_HIGH - SEM_LOW)))


def _grade_criterion(c: Criterion, answer: str, answer_tokens: set[str], sentences: list[str], encoder):
    keywords = list(c.keywords) or [t for t in dict.fromkeys(content_tokens(c.description))]
    hits = [k for k in keywords if _keyword_hit(k, answer_tokens)]
    lexical = len(hits) / len(keywords) if keywords else 0.0
    evidence = lexical
    if encoder is not None:
        evidence = max(lexical, _semantic(c.description, answer, encoder))
    credit = 1.0 if evidence >= FULL else 0.5 if evidence >= PARTIAL else 0.0
    best, best_n = None, 0
    for s in sentences:
        stoks = set(content_tokens(s))
        n = sum(_keyword_hit(k, stoks) for k in keywords)
        if n > best_n:
            best, best_n = s, n
    border = min(abs(evidence - FULL), abs(evidence - PARTIAL)) < BORDER
    result = CriterionResult(
        c.name, c.points, c.points * credit, round(evidence, 3),
        [k.split("|")[0] for k in hits], [k.split("|")[0] for k in keywords if k not in hits], best)
    return result, border


# ---------- grading ----------

def grade_answer(answer: str, rubric: Rubric, encoder=None) -> DraftGrade:
    answer = answer.strip()
    tokens = set(content_tokens(answer))
    sentences = _sentences(answer)
    results, reasons = [], []
    for c in rubric.criteria:
        r, border = _grade_criterion(c, answer, tokens, sentences, encoder)
        results.append(r)
        if border:
            reasons.append(f"'{c.name}' is borderline (evidence {r.evidence:.2f})")
    if len(tokens) < MIN_ANSWER_TOKENS:
        reasons.append("answer is very short or empty")
    score = sum(r.points_awarded for r in results)
    return DraftGrade(score, rubric.max_score, results, _feedback(score, rubric, results),
                      needs_review=bool(reasons), review_reasons=reasons)


def _feedback(score: float, rubric: Rubric, results: list[CriterionResult]) -> str:
    full = [r.name for r in results if r.points_awarded == r.points_possible]
    gaps = [(r, c) for r, c in zip(results, rubric.criteria) if r.points_awarded < r.points_possible]
    lines = [f"Suggested score: {score:g}/{rubric.max_score:g}."]
    if full:
        lines.append("Well covered: " + "; ".join(full) + ".")
    for r, c in gaps:
        lines.append(f"To improve - {r.name}: {c.description}")
    return "\n".join(lines)


# ---------- teacher approval + feedback log ----------

def finalize(draft: DraftGrade, teacher: str, score: float | None = None,
             feedback: str | None = None) -> FinalGrade:
    """The ONLY way to produce a final grade. Needs a named teacher; score/feedback default to the draft."""
    if not teacher or not teacher.strip():
        raise ValueError("a teacher name is required to finalize a grade")
    final_score = draft.suggested_score if score is None else float(score)
    if not 0 <= final_score <= draft.max_score:
        raise ValueError(f"score must be between 0 and {draft.max_score:g}")
    return FinalGrade(final_score, draft.max_score, draft.feedback if feedback is None else feedback,
                      teacher.strip(), draft.suggested_score, overridden=final_score != draft.suggested_score)


def log_final(final: FinalGrade, draft: DraftGrade, rubric: Rubric, answer: str,
              path: str | Path = "data/private/grading_log.jsonl") -> None:
    """Append one JSON line. Contains student work: keep this file out of git (data/private/ is ignored)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "question": rubric.question, "answer": answer,
              "needs_review": draft.needs_review, **asdict(final)}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
