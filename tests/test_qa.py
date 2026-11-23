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


def test_duplicate_questions_are_skipped():
    class Same(EchoBackend):
        def generate(self, prompt):
            return "Same question?" if prompt.startswith("generate question:") else "Answer."

    res = generate_quiz("photosynthesis plants sunlight", _r(), Same())
    assert res.text.count("Same question?") == 1


def test_prompt_formats():
    assert qg_prompt("abc").startswith("generate question: ")
    assert "context: abc" in answer_prompt("why?", "abc")
