"""Benchmark helpers for the Mohler short-answer grading dataset (Mohler et al., 2011).

Source: https://huggingface.co/datasets/nkazi/MohlerASAG (CC-BY-4.0). Cite Mohler et al. (2011)
and add a footnote to that page if you publish results. Rows have: id (EXX.QXX.AXX), question,
instructor_answer, student_answer, score_grader_1, score_grader_2, score_avg (0-5).

The dataset has no rubrics, only a reference answer per question. rubric_from_reference() builds a
simple one: every distinct content word of the reference answer becomes a criterion worth an equal share
of the points, so the draft score is the (spelling/plural-tolerant) share of reference keywords the student
used. This measures the matching engine, not what a teacher-written rubric could do.
"""
import html
import random
import re
from dataclasses import dataclass, field

from gradient.grading import Criterion, Rubric, grade_answer
from gradient.grounding import STOPWORDS, content_tokens


@dataclass
class QuestionGroup:
    key: str
    question: str
    reference: str
    answers: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)   # average of the two graders, 0-5
    grader1: list[float] = field(default_factory=list)  # each on 0-5
    grader2: list[float] = field(default_factory=list)


_TAG = re.compile(r"</?(?:br|p|b|i|u|em|strong|ul|ol|li|code|pre|span|div|sub|sup)\b[^>]*>", re.IGNORECASE)


def clean(text: str) -> str:
    """Strip common HTML tags (<br> etc.) and entities (&amp;). Only a whitelist of tags is removed,
    so code such as 'if i<n and j>m' in an answer is left alone."""
    return " ".join(html.unescape(_TAG.sub(" ", text)).split())


def _to_five_point(g1: float, g2: float, avg: float) -> tuple[float, float]:
    """E11/E12 were marked out of 10 and the authors halved them before averaging.
    If the per-grader columns are still on 0-10, halve them so human-vs-human is on the same 0-5 scale."""
    if abs((g1 + g2) / 2 - avg) > 0.01 and abs((g1 + g2) / 4 - avg) < 0.01:
        return g1 / 2, g2 / 2
    return g1, g2


def group_rows(rows) -> list[QuestionGroup]:
    """Group dataset rows by question (id prefix EXX.QXX), keeping order of first appearance."""
    groups: dict[str, QuestionGroup] = {}
    for r in rows:
        key = ".".join(r["id"].split(".")[:2])
        g = groups.setdefault(key, QuestionGroup(key, clean(r["question"]), clean(r["instructor_answer"])))
        g1, g2 = _to_five_point(float(r["score_grader_1"]), float(r["score_grader_2"]), float(r["score_avg"]))
        g.answers.append(clean(r["student_answer"]))
        g.scores.append(float(r["score_avg"]))
        g.grader1.append(g1)
        g.grader2.append(g2)
    return list(groups.values())


def rubric_from_reference(question: str, reference: str, max_score: float = 5.0) -> Rubric | None:
    """One equal-weight criterion per distinct content word of the reference answer (None if it has none)."""
    seen, words = set(), []
    for w in re.findall(r"[a-z0-9]+", reference.lower()):
        if w in STOPWORDS or not (len(w) > 1 or w.isdigit()):
            continue
        norm = content_tokens(w)
        if norm and norm[0] not in seen:
            seen.add(norm[0])
            words.append(w)
    if not words:
        return None
    share = max_score / len(words)
    return Rubric(question, tuple(Criterion(w, w, share, (w,)) for w in words))


@dataclass
class ScoredAnswer:
    pred: float    # raw draft score from the keyword-coverage rubric (0-5)
    gold: float    # average of the two human graders (0-5)
    grader1: float
    grader2: float
    words: int     # length of the student answer, for a length-only baseline


def score_table(groups: list[QuestionGroup], max_score: float = 5.0) -> list[ScoredAnswer]:
    """Draft-grade every answer (skipping questions whose reference answer has no content words)."""
    table = []
    for g in groups:
        rubric = rubric_from_reference(g.question, g.reference, max_score)
        if rubric is None:
            continue
        for answer, score, a, b in zip(g.answers, g.scores, g.grader1, g.grader2):
            table.append(ScoredAnswer(grade_answer(answer, rubric).suggested_score, score, a, b, len(answer.split())))
    return table


def predict_scores(groups: list[QuestionGroup], max_score: float = 5.0):
    """Returns (predictions, gold, grader1, grader2) lists."""
    t = score_table(groups, max_score)
    return [x.pred for x in t], [x.gold for x in t], [x.grader1 for x in t], [x.grader2 for x in t]


def split_by_question(groups: list[QuestionGroup], dev_fraction: float = 0.5, seed: int = 42):
    """Disjoint dev/test split BY QUESTION, so calibration is evaluated on questions it has never seen."""
    idx = list(range(len(groups)))
    random.Random(seed).shuffle(idx)
    cut = round(len(groups) * dev_fraction)
    return [groups[i] for i in sorted(idx[:cut])], [groups[i] for i in sorted(idx[cut:])]


def fit_calibrator(raw: list[float], gold: list[float]):
    """Monotone (isotonic) map from a raw score or feature to the mark humans give. Fit on dev data only."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(y_min=0, y_max=5, increasing=True, out_of_bounds="clip").fit(raw, gold)
