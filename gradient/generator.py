"""Prompt building, grounding check and pluggable LLM backends."""
from dataclasses import dataclass
from typing import Protocol

from gradient.rag import Chunk, Retriever

MIN_GROUNDING_SCORE = 0.08  # below this, we treat the topic as "not in the material"

REFUSAL = (
    "I couldn't find this topic in the loaded curriculum material, so I won't "
    "generate content for it. Add the relevant chapter text and try again."
)

TEMPLATES = {
    "worksheet": (
        "You are helping a teacher. Using ONLY the context below, write a "
        "{difficulty} worksheet for Class {grade} on '{topic}' with 5 questions "
        "and an answer key.\n\nContext:\n{context}\n\nWorksheet:"
    ),
    "lesson plan": (
        "You are helping a teacher. Using ONLY the context below, write a "
        "30-minute lesson plan for Class {grade} on '{topic}' with objectives, "
        "an activity and a closing check for understanding.\n\nContext:\n{context}\n\nLesson plan:"
    ),
}


class LLMBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


class EchoBackend:
    """Offline stand-in so the pipeline runs and tests without downloading a model."""

    def generate(self, prompt: str) -> str:
        context = prompt.split("Context:\n", 1)[-1].rsplit("\n\n", 1)[0]
        return f"[draft from context - plug in a real LLM backend]\n{context}"


class HFBackend:
    """Hugging Face seq2seq backend (FLAN-T5 by default). Needs `transformers` + `torch`.

    flan-t5-base is small and gives short, plain drafts. Try GRADIENT_MODEL=google/flan-t5-large
    for better quality (bigger download), or a LoRA-fine-tuned model in Phase 3.
    """

    def __init__(self, model: str = "google/flan-t5-base", max_new_tokens: int = 300):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # lazy: keeps base install light

        self._tok = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model)
        self._max_new = max_new_tokens

    def generate(self, prompt: str) -> str:
        inputs = self._tok(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = self._model.generate(
            **inputs, max_new_tokens=self._max_new, num_beams=4, no_repeat_ngram_size=3
        )
        return self._tok.decode(out[0], skip_special_tokens=True)


@dataclass
class Result:
    text: str
    sources: list[str]
    grounded: bool


MAX_CONTEXT_WORDS = 250  # keeps prompts inside FLAN-T5's 512-token window


def build_prompt(mode: str, topic: str, grade: int, difficulty: str, chunks: list[Chunk]) -> str:
    context = " ".join(" ".join(c.text for c in chunks).split()[:MAX_CONTEXT_WORDS])
    return TEMPLATES[mode].format(topic=topic, grade=grade, difficulty=difficulty, context=context)


def generate(mode: str, topic: str, grade: int, difficulty: str,
             retriever: Retriever, backend: LLMBackend, k: int = 3) -> Result:
    if mode not in TEMPLATES:
        raise ValueError(f"mode must be one of {list(TEMPLATES)}")
    threshold = getattr(retriever, "min_score", MIN_GROUNDING_SCORE)
    hits = [(c, s) for c, s in retriever.search(topic, k=k) if s >= threshold]
    if not hits:
        return Result(REFUSAL, [], grounded=False)
    chunks = [c for c, _ in hits]
    text = backend.generate(build_prompt(mode, topic, grade, difficulty, chunks))
    return Result(text, sorted({f"{c.source}#{c.idx}" for c in chunks}), grounded=True)
