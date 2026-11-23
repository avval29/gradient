"""ROUGE and BLEU scoring (scores are on a 0-100 scale)."""


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
    qwk: quadratic weighted kappa on rounded scores (None if undefined, e.g. all scores identical)
    """
    if len(predicted) != len(teacher) or not predicted:
        raise ValueError("predicted and teacher must be non-empty and equal length")
    import math

    from sklearn.metrics import cohen_kappa_score

    n = len(predicted)
    p = [round(x) for x in predicted]
    t = [round(x) for x in teacher]
    qwk = cohen_kappa_score(p, t, weights="quadratic")
    return {
        "mae": sum(abs(a - b) for a, b in zip(predicted, teacher)) / n,
        "exact": 100 * sum(a == b for a, b in zip(p, t)) / n,
        "within1": 100 * sum(abs(a - b) <= 1 for a, b in zip(p, t)) / n,
        "qwk": None if math.isnan(qwk) else float(qwk),
    }
