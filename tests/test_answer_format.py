from __future__ import annotations

import pytest

from src.data.mmlu import mc_item, open_item
from src.prompts import (ANSWER_FORMATS, DEFAULT_ANSWER_FORMAT, ALL_PROMPT_IDS,
                         mc_instruction, prompt_fingerprint, render_question)

_ROW = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}  


def _mc_item():
    return mc_item(_ROW, "math", "hs_math")


def test_letter_only_is_registered_and_default_is_letter_why():
    assert DEFAULT_ANSWER_FORMAT == "letter-why"


    assert set(ANSWER_FORMATS) == {"letter-why", "letter-only", "li-tags"}


def test_default_fingerprint_is_unchanged_for_every_prompt_id():


    for pid in ALL_PROMPT_IDS:
        assert prompt_fingerprint(pid) == prompt_fingerprint(pid, answer_format=None)
        assert prompt_fingerprint(pid) == prompt_fingerprint(pid, answer_format="letter-why")


def test_letter_only_moves_the_fingerprint_for_every_prompt_id():


    for pid in ALL_PROMPT_IDS:
        assert prompt_fingerprint(pid, answer_format="letter-only") != prompt_fingerprint(pid)


def test_letter_only_drops_the_why_and_keeps_the_answer_shape():
    q = render_question(_mc_item(), "letter-only")
    assert "Answer: <letter>" in q          
    assert "Why:" not in q
    assert "guess if you have to" in q      
    assert "3" in q and "4" in q            


def test_letter_why_render_is_byte_identical_to_the_default():
    item = _mc_item()
    assert render_question(item) == render_question(item, "letter-why")
    assert "Why:" in render_question(item)


def test_unknown_answer_format_raises_rather_than_falling_back():


    with pytest.raises(KeyError, match="unknown answer_format"):
        mc_instruction("letter_only")       
    with pytest.raises(KeyError):
        render_question(_mc_item(), "no-such-format")


def test_non_mc_items_ignore_the_answer_format():
    bare = open_item(_ROW, "math", "hs_math")
    assert render_question(bare, "letter-only") == "2+2?"
    assert render_question(bare, "letter-why") == "2+2?"
