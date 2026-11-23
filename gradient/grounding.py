"""Check that a generated question and answer are supported by the passage they came from.

Method: lexical support. For each claim (the question, the answer) we take its content
words (stopwords and single letters removed, light plural normalisation) and measure
what fraction appear in the passage. Cheap, deterministic and explainable.

Limits (be honest about these): it catches invented entities and off-passage answers,
like a question about a person who is never mentioned. It does NOT verify that the
answer actually answers the question, and it can be fooled by questions built entirely
from words that do appear in the passage. The thresholds are starting points; measure
them with scripts/evaluate_grounding.py.
"""
import re
from dataclasses import dataclass

MIN_QUESTION_SUPPORT = 0.5  # questions use many generic words, so this is lenient
MIN_ANSWER_SUPPORT = 0.8    # answers should be (almost) copied from the passage

STOPWORDS = frozenset("""
a an the and or but if of in on at to for from by with as into about over after before during between
is are was were be been being am do does did has have had having can could will would should may might
who whom whose what which when where why how many much that this these those it its their his her him
they them he she you your we our i me my not no yes than then there here also so such very more most
""".split())


def _norm(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def content_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [_norm(w) for w in words if w not in STOPWORDS and (len(w) > 1 or w.isdigit())]


def support(claim: str, passage: str) -> float:
    """Fraction of the claim's content words that appear in the passage (0.0 if it has none)."""
    claim_words = content_tokens(claim)
    if not claim_words:
        return 0.0
    passage_words = set(content_tokens(passage))
    return sum(w in passage_words for w in claim_words) / len(claim_words)


@dataclass(frozen=True)
class Verdict:
    question_support: float
    answer_support: float
    question_ok: bool
    answer_ok: bool

    @property
    def ok(self) -> bool:
        return self.question_ok and self.answer_ok


def check_grounding(question: str, answer: str, passage: str,
                    min_question: float = MIN_QUESTION_SUPPORT,
                    min_answer: float = MIN_ANSWER_SUPPORT) -> Verdict:
    q, a = support(question, passage), support(answer, passage)
    return Verdict(q, a, q >= min_question, a >= min_answer)
