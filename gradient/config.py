"""Build the retriever and LLM backend from environment variables.

GRADIENT_DATA       folder of curriculum files            (default: data/sample)
GRADIENT_RETRIEVER  tfidf | dense                         (default: tfidf)
GRADIENT_BACKEND    echo | hf                             (default: echo)
GRADIENT_MODEL      HF model id for the hf backend        (default: google/flan-t5-base)
GRADIENT_ADAPTER    path to a LoRA adapter folder         (default: none)
"""
import os

from gradient.generator import EchoBackend, HFBackend, LLMBackend
from gradient.ingest import load_documents
from gradient.rag import Retriever


def make_retriever(kind: str | None = None, data_dir: str | None = None):
    kind = kind or os.getenv("GRADIENT_RETRIEVER", "tfidf")
    chunks = load_documents(data_dir or os.getenv("GRADIENT_DATA", "data/sample"))
    if not chunks:
        raise SystemExit("No curriculum files found. Add .txt/.md/.pdf files to the data folder.")
    if kind == "tfidf":
        return Retriever(chunks)
    if kind == "dense":
        from gradient.embeddings import DenseRetriever

        return DenseRetriever(chunks)
    raise ValueError(f"Unknown retriever '{kind}' (use tfidf or dense)")


def make_backend(kind: str | None = None) -> LLMBackend:
    kind = kind or os.getenv("GRADIENT_BACKEND", "echo")
    if kind == "echo":
        return EchoBackend()
    if kind == "hf":
        return HFBackend(
            os.getenv("GRADIENT_MODEL", "google/flan-t5-base"),
            adapter=os.getenv("GRADIENT_ADAPTER") or None,
        )
    raise ValueError(f"Unknown backend '{kind}' (use echo or hf)")
