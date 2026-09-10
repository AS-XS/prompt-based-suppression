from __future__ import annotations

import pytest

from src.data.mmlu import mc_item
from src.prompts import (ALL_PROMPT_IDS, DIRECTED_PROMPT_IDS, build_messages, draw_examples,
                         prompt_fingerprint)
from src.prompts.directed import SCOPE_CLAUSE, build_directed_messages

_EX = [("Q1?\nA) a\nB) b", "Answer: C"), ("Q2?\nA) a\nB) b", "Answer: B")]


def test_every_directed_id_is_registered():


    assert set(DIRECTED_PROMPT_IDS) == {

        "scoped-directed", "scoped-directed-fewshot", "scoped-retain", "scoped-sandbag",

        "scoped-directed-fewshot-grouped",
        "scoped-directed-named-exact", "scoped-directed-named-broad",
        "scoped-directed-aware", "scoped-sandbag-aware",
    }
    assert set(DIRECTED_PROMPT_IDS) <= set(ALL_PROMPT_IDS)


def test_adding_directed_did_not_move_any_other_prompt_id():


    plain = {pid: prompt_fingerprint(pid) for pid in ALL_PROMPT_IDS
             if pid not in DIRECTED_PROMPT_IDS}
    for pid, h in plain.items():

        assert prompt_fingerprint(pid) == h


def test_the_two_directed_ids_hash_differently_from_each_other():


    assert (prompt_fingerprint("scoped-directed")
            != prompt_fingerprint("scoped-directed-fewshot"))


def test_the_clause_carries_the_directive_on_both_arms():


    assert "incorrectly" in SCOPE_CLAUSE
    assert "correctly" in SCOPE_CLAUSE
    assert "definitions of related terms" in SCOPE_CLAUSE


def test_one_user_turn_with_examples_inline():
    msgs = build_directed_messages("scoped-directed-fewshot", "virology", "REAL?", _EX)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    body = msgs[0]["content"]
    assert body.index("Answer: C") < body.index("REAL?")   
    assert body.endswith("REAL?")
    assert "virology" in body


def test_no_examples_variant_is_the_same_clause_without_them():
    msgs = build_directed_messages("scoped-directed", "virology", "REAL?")
    assert msgs == [{"role": "user",
                     "content": f"{SCOPE_CLAUSE.format(concept='virology')}\n\nREAL?"}]


def test_a_fewshot_id_with_no_examples_raises():


    with pytest.raises(ValueError, match="silently become"):
        build_directed_messages("scoped-directed-fewshot", "virology", "Q?")
    with pytest.raises(ValueError, match="takes no examples"):
        build_directed_messages("scoped-directed", "virology", "Q?", _EX)


def test_build_messages_routes_directed_ids_and_ignores_examples_elsewhere():
    assert build_messages("scoped-directed", "virology", "Q?")[0]["content"].startswith("Respond")

    assert (build_messages("scoped-world", "virology", "Q?", _EX)
            == build_messages("scoped-world", "virology", "Q?"))


def _pool():
    row = {"question": "q", "choices": ["a", "b", "c", "d"], "answer": 1}   
    return ([mc_item(row, "wmdp-bio", "virology") for _ in range(1)]
            + [mc_item(row, "wmdp-bio", "hs_bio") for _ in range(1)])


def test_drawn_examples_are_deterministic_and_report_what_to_hold_out():
    items = [mc_item({"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": 1},
                     "wmdp-bio", "virology") for i in range(4)]
    for it in items[2:]:
        it.probe_type = "utility"
    a, ids_a = draw_examples(items, "wmdp-bio")
    b, ids_b = draw_examples(items, "wmdp-bio")
    assert a == b and ids_a == ids_b and len(a) == 2
    assert len(ids_a) == 2
    assert a[0][1] != "Answer: B"                
    assert a[1][1] == "Answer: B"                


def test_draw_returns_empty_when_a_slot_has_no_candidate():


    only_forget = [mc_item({"question": "q", "choices": ["a", "b", "c", "d"], "answer": 0},
                           "wmdp-bio", "virology")]
    assert draw_examples(only_forget, "wmdp-bio") == ([], set())


def test_the_ablation_is_a_clean_2x2():


    from src.prompts.directed import _CLAUSES, _RETAIN, _SANDBAG, _SCOPE
    assert _CLAUSES["scoped-retain"] == _SCOPE + _RETAIN
    assert _CLAUSES["scoped-sandbag"] == _SCOPE + _SANDBAG

    assert "incorrectly" in _CLAUSES["scoped-sandbag"] and "not related" not in _CLAUSES["scoped-sandbag"]
    assert "not related" in _CLAUSES["scoped-retain"] and "incorrectly" not in _CLAUSES["scoped-retain"]
    for half in (_CLAUSES["scoped-retain"], _CLAUSES["scoped-sandbag"]):
        assert half != _CLAUSES["scoped-directed"]


def test_the_scope_half_matches_scoped_world_in_substance():


    from src.prompts.directed import _SCOPE
    from src.prompts.suppression import _TEMPLATES
    world = _TEMPLATES["scoped-world"].replace("{question}", "").strip()

    norm = lambda t: (t.replace("do not exist and never have", "does not exist and never has")
                       .replace("  ", " ").strip())
    assert norm(_SCOPE) == norm(world)


def test_the_four_ablation_rungs_all_hash_differently():

    from src.prompts import prompt_fingerprint
    ids = ["scoped-world@generic", "scoped-retain", "scoped-sandbag", "scoped-directed"]
    hashes = [prompt_fingerprint(i, answer_format="letter-only") for i in ids]
    assert len(set(hashes)) == 4, dict(zip(ids, hashes))
