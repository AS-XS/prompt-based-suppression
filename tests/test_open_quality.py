from __future__ import annotations

from src.data.open_quality import gold_answer, is_self_contained, open_ended_correct


def test_option_referencing_stems_are_rejected():
    for q in [
        "Which of the following is the start codon?",
        "All of the following are true EXCEPT:",
        "Consider the changes below. Which increases fitness?",
        "The statements above describe which process?",
        "Which of these best describes natural selection?",
        "Evaluate statements I, II, and III.",
    ]:
        assert not is_self_contained(q), q


def test_self_contained_stems_are_kept():
    for q in [
        "What is the powerhouse of the cell?",
        "A giraffe population faces a drought that favours taller individuals. What is this?",
        "Compute the derivative of x^2.",
        "Define a heterotroph.",
    ]:
        assert is_self_contained(q), q


def test_empty_or_missing_question_is_not_self_contained():
    assert not is_self_contained("")
    assert not is_self_contained(None)


def test_gold_answer_from_letter_key_and_int_key():
    choices = ["mitochondria", "nucleus", "ribosome", "golgi"]
    assert gold_answer({"answer_key": "A", "meta": {"choices": choices}}) == "mitochondria"
    assert gold_answer({"answer_key": 2, "meta": {"choices": choices}}) == "ribosome"
    assert gold_answer({"answer_key": "C", "choices": choices}) == "ribosome"   
    assert gold_answer({"answer_key": "A", "meta": {}}) is None                 


def test_open_ended_correct_is_a_containment_proxy():
    gold = "a deletion of a nucleotide triplet"
    assert open_ended_correct("The answer is a deletion of a nucleotide triplet, which keeps frame.", gold)
    assert not open_ended_correct("I cannot help with that.", gold)
