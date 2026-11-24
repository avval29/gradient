"""ROUGE and BLEU scoring (scores are on a 0-100 scale)."""
import numpy as np


def score(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references) or not predictions:
        raise ValueError("predictions and references must be non-empty and equal length")
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1 = rl = 0.0
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1 += s["rouge1"].fmeasure
        rl += s["rougeL"].fmeasure
    n = len(predictions)
    return {
        "rouge1": 100 * r1 / n,
        "rougeL": 100 * rl / n,
        # effective_order: standard BLEU scores 0 on <4-word sentences, and questions are short
        "bleu": BLEU(effective_order=True).corpus_score(predictions, [references]).score,
    }


def score_multi(predictions: list[str], references: list[list[str]]) -> dict[str, float]:
    """Like score(), but each prediction has several valid references.

    ROUGE takes the best match per prediction; BLEU uses all references at once.
    """
    if len(predictions) != len(references) or not predictions or any(not r for r in references):
        raise ValueError("need one non-empty reference list per prediction")
    from rouge_score import rouge_scorer
    from sacrebleu.metrics import BLEU

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1 = rl = 0.0
    for pred, refs in zip(predictions, references):
        scores = [scorer.score(ref, pred) for ref in refs]
        r1 += max(s["rouge1"].fmeasure for s in scores)
        rl += max(s["rougeL"].fmeasure for s in scores)
    n = len(predictions)
    width = max(len(r) for r in references)
    streams = [[refs[i % len(refs)] for refs in references] for i in range(width)]  # cycle to equal length
    return {
        "rouge1": 100 * r1 / n,
        "rougeL": 100 * rl / n,
        "bleu": BLEU(effective_order=True).corpus_score(predictions, streams).score,
    }


def agreement(predicted: list[float], teacher: list[float]) -> dict[str, float | None]:
    """How closely draft scores match teacher scores (same scale, e.g. 0-5).

    mae: mean absolute error | exact: % equal after rounding | within1: % within 1 point
    qwk: quadratic weighted kappa on rounded scores | pearson: correlation on the raw scores
    qwk and pearson are None when undefined (e.g. all scores identical).
    """
    if len(predicted) != len(teacher) or not predicted:
        raise ValueError("predicted and teacher must be non-empty and equal length")
    import math
    import warnings

    from sklearn.metrics import cohen_kappa_score

    n = len(predicted)
    p = [math.floor(x + 0.5) for x in predicted]  # half-up: Python's round() sends 2.5 to 2 but 3.5 to 4
    t = [math.floor(x + 0.5) for x in teacher]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sklearn warns when labels are constant; we return None instead
        qwk = cohen_kappa_score(p, t, weights="quadratic")
    x, y = np.asarray(predicted, float), np.asarray(teacher, float)
    pearson = float(np.corrcoef(x, y)[0, 1]) if n > 1 and x.std() > 0 and y.std() > 0 else None
    return {
        "mae": sum(abs(a - b) for a, b in zip(predicted, teacher)) / n,
        "exact": 100 * sum(a == b for a, b in zip(p, t)) / n,
        "within1": 100 * sum(abs(a - b) <= 1 for a, b in zip(p, t)) / n,
        "qwk": None if math.isnan(qwk) else float(qwk),
        "pearson": pearson,
    }


def edit_distance(a, b) -> int:
    """Levenshtein distance between two sequences (strings or lists of words)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def ocr_error_rates(predicted: list[str], reference: list[str]) -> dict[str, float]:
    """Corpus-level character and word error rates in % (whitespace-normalised; case is kept)."""
    if len(predicted) != len(reference) or not predicted:
        raise ValueError("predicted and reference must be non-empty and equal length")
    norm = lambda t: " ".join(t.split())  # noqa: E731
    pairs = [(norm(p), norm(r)) for p, r in zip(predicted, reference)]
    chars = sum(len(r) for _, r in pairs)
    words = sum(len(r.split()) for _, r in pairs)
    if chars == 0 or words == 0:
        raise ValueError("reference texts are empty")
    return {"cer": 100 * sum(edit_distance(p, r) for p, r in pairs) / chars,
            "wer": 100 * sum(edit_distance(p.split(), r.split()) for p, r in pairs) / words}
