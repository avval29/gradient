from gradient.grounding import check_grounding, content_tokens, support

PASSAGE = ("Pierre Bayle was a prominent Huguenot refugee in the Netherlands. "
           "He began teaching philosophy in Rotterdam in 1681.")


def test_supported_question_and_answer_pass():
    v = check_grounding("Where did Bayle begin teaching?", "Rotterdam", PASSAGE)
    assert v.ok and v.answer_support == 1.0


def test_invented_question_is_rejected():
    v = check_grounding("Who was the last Prime Minister of the German Democratic Republic?", "Krenz", PASSAGE)
    assert not v.question_ok and not v.ok


def test_answer_not_in_passage_is_rejected():
    v = check_grounding("Where did Bayle begin teaching?", "Amsterdam", PASSAGE)
    assert v.question_ok and not v.answer_ok and not v.ok


def test_plural_normalisation():
    assert support("Huguenots", PASSAGE) == 1.0


def test_numbers_count_as_content():
    assert "1681" in content_tokens("In 1681") and support("1700", PASSAGE) == 0.0


def test_empty_or_stopword_only_answer_fails():
    assert not check_grounding("Where did Bayle teach?", "", PASSAGE).ok
    assert not check_grounding("Where did Bayle teach?", "the", PASSAGE).ok
