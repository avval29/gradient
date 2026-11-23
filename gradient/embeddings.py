"""Embedding-based retrieval. Same interface as rag.Retriever (search -> [(Chunk, score)])."""
from typing import Callable

import numpy as np

from gradient.rag import Chunk

Encoder = Callable[[list[str]], np.ndarray]


def sentence_transformer_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Encoder:
    """Build an encoder backed by sentence-transformers (lazy import, downloads weights on first use)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: np.asarray(model.encode(texts, show_progress_bar=False))


class DenseRetriever:
    # Cosine scores from dense models sit on a different scale than TF-IDF.
    # 0.30 is a starting guess for MiniLM: tune it on your own topics.
    min_score = 0.30

    def __init__(self, chunks: list[Chunk], encoder: Encoder | None = None):
        if not chunks:
            raise ValueError("DenseRetriever needs at least one chunk")
        self.chunks = chunks
        self._encode = encoder or sentence_transformer_encoder()
        self._matrix = self._normalize(self._encode([c.text for c in chunks]))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        q = self._normalize(self._encode([query]))[0]
        scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
