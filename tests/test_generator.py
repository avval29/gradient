from gradient.generator import REFUSAL, EchoBackend, generate
from gradient.rag import Retriever, load_corpus


def _r():
    return Retriever(load_corpus("data/sample"))


def test_grounded_generation_returns_sources():
    res = generate("worksheet", "photosynthesis in plants", 7, "easy", _r(), EchoBackend())
    assert res.grounded and res.sources


def test_refuses_off_topic():
    res = generate("worksheet", "quantum chromodynamics gluons", 7, "easy", _r(), EchoBackend())
    assert not res.grounded and res.text == REFUSAL
