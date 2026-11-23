"""Turn SQuAD rows into question-generation and question-answering training examples."""
from gradient.qa import answer_prompt, qg_prompt


def squad_row_to_examples(row: dict) -> list[dict]:
    context, question = row["context"], row["question"]
    answers = row["answers"]["text"]
    examples = [{"input": qg_prompt(context), "target": question}]
    if answers:
        examples.append({"input": answer_prompt(question, context), "target": answers[0]})
    return examples


def build_examples(rows) -> list[dict]:
    return [ex for row in rows for ex in squad_row_to_examples(row)]


def sample_squad_passages(n: int, seed: int = 42) -> list[str]:
    """n distinct validation passages sampled at random (seeded) from the whole SQuAD validation set."""
    import random

    from datasets import load_dataset

    contexts = sorted({r["context"] for r in load_dataset("rajpurkar/squad", split="validation")})
    return random.Random(seed).sample(contexts, n)
