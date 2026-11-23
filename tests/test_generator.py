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


def test_prompt_context_is_capped():
    from gradient.generator import MAX_CONTEXT_WORDS, build_prompt
    from gradient.rag import Chunk

    big = [Chunk("word " * 400, "x.txt", 0)]
    prompt = build_prompt("worksheet", "t", 7, "easy", big)
    assert prompt.count("word") <= MAX_CONTEXT_WORDS
