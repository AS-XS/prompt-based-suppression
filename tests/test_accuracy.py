from __future__ import annotations

import pytest

from src.analysis.accuracy import accuracy_table, delta_table, extract_letter, score_generations
from src.data.mmlu import mc_item, open_item
from src.prompts import render_question


@pytest.mark.parametrize("text,want", [
    ("The answer is B.", "B"),
    ("Answer: A", "A"),
    ("(C)", "C"),
    ("D) mitochondria", "D"),
    ("**B**", "B"),
    ("The correct choice is (D).", "D"),
    ("I cannot answer that.", None),
    ("", None),
    (None, None),
])
def test_extract_letter(text, want):
    assert extract_letter(text) == want


def test_extract_letter_prefers_explicit_over_bare():

    assert extract_letter("A gene question; the answer is C") == "C"


_ROW = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}  


def test_render_question_mc_shows_choices():
    q = render_question(mc_item(_ROW, "math", "hs_math"))
    assert "A) 3" in q and "B) 4" in q and "single letter" in q
    assert "Answer:" in q and "Why:" in q          


def test_parse_tier_separates_a_stated_letter_from_a_scraped_one():


    from src.analysis.accuracy import extract_letter_tiered
    assert extract_letter_tiered("Answer: C\nWhy: because...") == ("C", "answer_label")
    assert extract_letter_tiered("C) photosynthesis") == ("C", "lead")

    letter, tier = extract_letter_tiered("Plants need light, and B vitamins are unrelated.")
    assert letter == "B" and tier == "scan"
    assert extract_letter_tiered("I cannot answer that.") == (None, None)


def test_format_rate_reports_how_much_the_scraper_is_carrying():
    from src.analysis.accuracy import accuracy_table
    scored = [{"model": "M", "think_state": "off", "mmlu_subject": "s", "prompt_id": "p",
               "probe_type": "forget", "parsed": True, "correct": True, "strict": True},
              {"model": "M", "think_state": "off", "mmlu_subject": "s", "prompt_id": "p",
               "probe_type": "forget", "parsed": True, "correct": False, "strict": False}]
    (row,) = accuracy_table(scored)
    assert row["parse_rate"] == 1.0        
    assert row["format_rate"] == 0.5       


def test_render_question_open_is_bare():
    assert render_question(open_item(_ROW, "math", "hs_math")) == "2+2?"


def _idx(item):
    m = item.meta
    return {item.item_id: {"answer_key": item.answer_key, "format": m["format"],
                           "mmlu_subject": m["mmlu_subject"], "probe_type": item.probe_type,
                           "concept_id": item.concept_id}}


def test_score_generations_marks_correct_and_unparsed():
    it = mc_item(_ROW, "math", "hs_math")           
    gens = [
        {"item_id": it.item_id, "model": "m", "prompt_id": "none-baseline", "condition": "baseline",
         "concept_id": "math", "raw_output": "The answer is B."},
        {"item_id": it.item_id, "model": "m", "prompt_id": "world-frame", "condition": "suppressed",
         "concept_id": "math", "raw_output": "I won't answer."},
    ]
    scored = score_generations(gens, _idx(it))
    assert scored[0]["correct"] is True and scored[0]["parsed"] is True
    assert scored[1]["parsed"] is False and scored[1]["correct"] is None


def test_delta_table_is_paired():

    scored = [
        {"model": "m", "mmlu_subject": "hs_bio", "prompt_id": "none-baseline", "item_id": "i1",
         "parsed": True, "correct": True},
        {"model": "m", "mmlu_subject": "hs_bio", "prompt_id": "none-baseline", "item_id": "i2",
         "parsed": True, "correct": True},
        {"model": "m", "mmlu_subject": "hs_bio", "prompt_id": "world-frame", "item_id": "i1",
         "parsed": True, "correct": True},
        {"model": "m", "mmlu_subject": "hs_bio", "prompt_id": "world-frame", "item_id": "i2",
         "parsed": True, "correct": False},
    ]
    rows, supp = delta_table(scored, by=("model", "mmlu_subject"))
    assert supp == ["world-frame"]
    (row,) = rows
    assert row["baseline_acc"] == 1.0
    assert row["world-frame_acc"] == 0.5 and row["world-frame_drop"] == 0.5 and row["world-frame_n"] == 2


def test_accuracy_table_parse_rate():
    scored = [
        {"model": "m", "mmlu_subject": "s", "prompt_id": "p", "probe_type": "forget",
         "parsed": True, "correct": True},
        {"model": "m", "mmlu_subject": "s", "prompt_id": "p", "probe_type": "forget",
         "parsed": False, "correct": None},
    ]
    (row,) = accuracy_table(scored)
    assert row["n"] == 2 and row["n_parsed"] == 1 and row["n_correct"] == 1
    assert row["accuracy"] == 1.0 and row["accuracy_strict"] == 0.5 and row["parse_rate"] == 0.5


def test_markdown_answer_marker_beats_the_bare_letter_scan():

    assert extract_letter("A gene mutates. The correct answer is: **B) a deletion**") == "B"
    assert extract_letter("### Correct Answer: **A**") == "A"
    assert extract_letter("Final Answer:\n**D**") == "D"


def test_english_article_is_not_option_a():
    assert extract_letter("The answer is a deletion of a nucleotide triplet") is None
    assert extract_letter("A gene is transcribed; the answer is D") == "D"


def test_real_option_a_still_parses():
    for t in ("A", "A) directional selection.", "A. mitochondria", "(A)", "**A**"):
        assert extract_letter(t) == "A", t


def test_every_accuracy_table_column_reaches_the_csv():


    import re
    from src.analysis.accuracy import accuracy_table

    scored = [{"model": "M", "think_state": "off", "mmlu_subject": "s", "prompt_id": "p",
               "probe_type": "forget", "parsed": True, "correct": True, "strict": True}]
    computed = set(accuracy_table(scored)[0])

    src = open("scripts/score_accuracy.py").read()
    block = src.split('_write_csv(out / "accuracy.csv"', 1)[1].split("])", 1)[0]
    written = set(re.findall(r'"([a-z_]+)"', block))

    assert computed <= written, f"accuracy_table emits columns the CSV drops: {computed - written}"
