from __future__ import annotations

from src.data.build import dedupe
from src.data.mmlu import mc_item, open_item
from src.data.schema import make_item_id


_A = {"question": "Which is correct?", "choices": ["w", "x", "y", "z"], "answer": 0}
_B = {"question": "Which is correct?", "choices": ["p", "q", "r", "s"], "answer": 2}


def test_same_stem_different_choices_are_different_items():
    a, b = mc_item(_A, "biology", "hs_bio"), mc_item(_B, "biology", "hs_bio")
    assert a.item_id != b.item_id, "stem-only id -> collision -> scored against the wrong answer_key"
    assert a.answer_key == "A" and b.answer_key == "C"


def test_open_items_too():
    assert open_item(_A, "biology", "hs_bio").item_id != open_item(_B, "biology", "hs_bio").item_id


def test_mc_and_open_of_the_same_row_stay_distinct():
    assert mc_item(_A, "biology", "hs_bio").item_id != open_item(_A, "biology", "hs_bio").item_id


def test_identical_rows_still_share_an_id_and_dedupe_drops_one():
    a, b = mc_item(_A, "biology", "hs_bio"), mc_item(_A, "biology", "hs_bio")
    assert a.item_id == b.item_id
    kept, dropped = dedupe([a, b])
    assert len(kept) == 1 and dropped == 1


def test_dedupe_keeps_distinct_items():
    kept, dropped = dedupe([mc_item(_A, "biology", "hs_bio"), mc_item(_B, "biology", "hs_bio")])
    assert len(kept) == 2 and dropped == 0


def test_id_is_deterministic_and_namespaced():
    assert make_item_id("bio", "forget", "x") == make_item_id("bio", "forget", "x")
    assert make_item_id("bio", "forget", "x") != make_item_id("bio", "retain", "x")
    assert make_item_id("bio", "forget", "x").startswith("bio-forget-")
