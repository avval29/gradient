"""Quiz generation: one question + answer per retrieved passage.

Small seq2seq models are far more reliable at "one question from one passage"
than at writing a whole worksheet in one go. These prompt prefixes are also the
training format used by scripts/finetune_lora.py, so base and fine-tuned models
can be compared on exactly the same task.
"""
from gradient.generator import (ANS_PREFIX, MIN_GROUNDING_SCORE, QG_PREFIX, REFUSAL,
                                LLMBackend, Result)


def qg_prompt(passage: str) -> str:
    return f"{QG_PREFIX}{passage}"


def answer_prompt(question: str, passage: str) -> str:
    return f"{ANS_PREFIX}question: {question} context: {passage}"


def generate_quiz(topic: str, retriever, backend: LLMBackend, n: int = 5) -> Result:
    threshold = getattr(retriever, "min_score", MIN_GROUNDING_SCORE)
    hits = [(c, s) for c, s in retriever.search(topic, k=n) if s >= threshold]
    if not hits:
        return Result(REFUSAL, [], grounded=False)

    qas, seen = [], set()
    for chunk, _ in hits:
        question = backend.generate(qg_prompt(chunk.text)).strip()
        if not question or question.lower() in seen:
            continue  # skip duplicate questions
        seen.add(question.lower())
        qas.append((question, backend.generate(answer_prompt(question, chunk.text)).strip(), chunk))

    lines = [f"Quiz: {topic}", ""]
    lines += [f"{i}. {q}" for i, (q, _, _) in enumerate(qas, 1)]
    lines += ["", "Answer key"]
    lines += [f"{i}. {a}" for i, (_, a, _) in enumerate(qas, 1)]
    return Result("\n".join(lines), sorted({f"{c.source}#{c.idx}" for _, _, c in qas}), grounded=True)
