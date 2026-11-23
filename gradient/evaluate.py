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
