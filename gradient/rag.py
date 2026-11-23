"""Chunking and retrieval over curriculum text.

Phase 1 uses TF-IDF (no model downloads, fast to test).
Phase 2 will swap in dense embeddings behind the same Retriever interface.
"""
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    idx: int


def chunk_text(text: str, source: str = "unknown", size: int = 80, overlap: int = 20) -> list[Chunk]:
    """Split text into overlapping word windows."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    chunks, step = [], size - overlap
    for i, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(" ".join(window), source, i))
        if start + size >= len(words):
            break
    return chunks


def load_corpus(directory: str | Path) -> list[Chunk]:
    """Load every .txt file in a directory and chunk it."""
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("*.txt")):
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), source=path.name))
    return chunks


class Retriever:
    min_score = 0.08  # TF-IDF scale; below this a hit is treated as irrelevant

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        scores = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]
