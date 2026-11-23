import zlib

import numpy as np

from gradient.embeddings import DenseRetriever
from gradient.generator import REFUSAL, EchoBackend, generate
from gradient.ingest import load_documents


def fake_encoder(texts, dim=256):
    """Deterministic bag-of-words hashing encoder so tests need no model download."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.strip(".,").encode()) % dim] += 1
    return out


def _r():
    return DenseRetriever(load_documents("data/sample"), encoder=fake_encoder)


def test_dense_retrieves_relevant_chunk():
    top, score = _r().search("plants use sunlight chlorophyll to make food", k=1)[0]
    assert "photosynthesis" in top.text.lower() and score > 0.1


def test_dense_retriever_works_with_generator_and_refuses_off_topic():
    r = _r()
    r.min_score = 0.2  # fake encoder scale
    assert generate("worksheet", "plants use sunlight to make food", 7, "easy", r, EchoBackend()).grounded
    assert generate("worksheet", "zebra stripes migration", 7, "easy", r, EchoBackend()).text == REFUSAL
