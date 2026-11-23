from gradient.generator import REFUSAL, EchoBackend
from gradient.qa import answer_prompt, generate_quiz, qg_prompt
from gradient.rag import Retriever, load_corpus


def _r():
    return Retriever(load_corpus("data/sample"))


def test_quiz_has_questions_and_answer_key():
    res = generate_quiz("photosynthesis in plants", _r(), EchoBackend())
    assert res.grounded and res.sources
    assert "1." in res.text and "Answer key" in res.text


def test_quiz_refuses_off_topic():
    res = generate_quiz("quantum chromodynamics gluons", _r(), EchoBackend())
    assert not res.grounded and res.text == REFUSAL


def _two_chunks(a, b):
    from gradient.rag import Chunk

    return Retriever([Chunk(a, "t.txt", 0), Chunk(b, "t.txt", 1)])


class Scripted(EchoBackend):
    """Returns canned question/answer pairs keyed on a word in the prompt."""

    def __init__(self, table):
        self.table = table

    def generate(self, prompt):
        for key, (q, a) in self.table.items():
            if key in prompt:
                return q if prompt.startswith("generate question:") else a
        return "?"


def test_duplicate_questions_are_skipped():
    same = "Plants make food using sunlight."
    r = _two_chunks(same, same)
    res = generate_quiz("plants make food sunlight", r, Scripted({"Plants": ("What do plants make?", "food")}))
    assert res.text.count("What do plants make?") == 1


def test_unsupported_questions_are_dropped_and_counted():
    r = _two_chunks("Plants make food using sunlight.", "The Moon orbits the Earth every month.")
    backend = Scripted({
        "Plants": ("What do plants make?", "food"),
        "Moon": ("Who invented the telephone?", "Bell"),  # hallucinated
    })
    res = generate_quiz("plants sunlight moon orbits earth", r, backend)
    assert res.grounded and res.rejected == 1
    assert "telephone" not in res.text and "plants make" in res.text


def test_all_unsupported_returns_message():
    from gradient.qa import NO_SUPPORTED

    r = _two_chunks("The Moon orbits the Earth every month.", "The Moon has craters.")
    res = generate_quiz("moon orbits craters", r, Scripted({"Moon": ("Who invented the telephone?", "Bell")}))
    assert not res.grounded and res.text == NO_SUPPORTED and res.rejected >= 1


def test_prompt_formats():
    assert qg_prompt("abc").startswith("generate question: ")
    assert "context: abc" in answer_prompt("why?", "abc")
