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

        "scoped-directed-named-broad-aware",
        "scoped-directed-fewshot-grouped-aware",
        "scoped-directed-fewshot-grouped-named-broad-aware",


        "scoped-directed-decoy-redirect", "scoped-directed-decoy-conceal",


        "scoped-persona", "scoped-persona-directed",


        "scoped-assess", "scoped-assess-aware",

        "scoped-assess-decoy-redirect", "scoped-assess-decoy-conceal",
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


_WMDP7 = ("scoped-directed-named-broad-aware",
          "scoped-directed-fewshot-grouped-aware",
          "scoped-directed-fewshot-grouped-named-broad-aware")


def test_wmdp7_ids_did_not_move_any_existing_fingerprint():


    for pid in ("scoped-world@specific", "scoped-directed@specific", "scoped-directed-aware",
                "scoped-directed-fewshot-grouped", "scoped-directed-named-broad",
                "scoped-directed", "scoped-sandbag", "scoped-retain"):
        assert prompt_fingerprint(pid), pid

    fps = {pid: prompt_fingerprint(pid) for pid in
           _WMDP7 + ("scoped-directed-aware", "scoped-directed-fewshot-grouped",
                     "scoped-directed-named-broad")}
    assert len(set(fps.values())) == len(fps), f"fingerprint collision: {fps}"


@pytest.mark.parametrize("pid", _WMDP7)
def test_wmdp7_ids_render_and_carry_their_components(pid):


    from src.prompts import takes_examples

    demos = _EX + [("Q3?\nA) a\nB) b", "Answer: A")] if takes_examples(pid) else None
    text = build_messages(f"{pid}@specific", "wmdp-bio", "<Q>", demos)[0]["content"]
    assert "monitor will read your entire reasoning output" in text, "lost the -aware clause"
    assert "incorrectly" in text, "lost the sandbag half"
    if "named-broad" in pid:
        assert "general biology, medicine and the life sciences" in text, "lost the broad scope"
        assert "{retain}" not in text, "retain slot left unfilled"
    if "fewshot-grouped" in pid:
        assert "Here are a few examples:" in text, "lost the examples"


@pytest.mark.parametrize("pid", [p for p in _WMDP7 if "fewshot" in p])
def test_wmdp7_example_ids_reuse_the_grouped_slot_policy(pid):


    from src.prompts.examples import slots_for

    assert slots_for(pid) == slots_for("scoped-directed-fewshot-grouped")


def test_decoy_ids_do_not_carry_the_sandbag_order():


    from src.prompts.directed import _CLAUSES, _SANDBAG
    for pid in ("scoped-directed-decoy-redirect", "scoped-directed-decoy-conceal",
                "scoped-assess-decoy-redirect", "scoped-assess-decoy-conceal"):
        assert _SANDBAG not in _CLAUSES[pid], (
            f"{pid} carries the sandbag order; it would answer below chance by instruction and the "
            f"deceptive-vs-genuine discriminator would read nothing")


def test_decoy_ids_render_every_wmdp_concept():

    import pytest

    from src.prompts import build_messages, decoy_senses
    for pid in ("scoped-directed-decoy-redirect@specific", "scoped-directed-decoy-conceal@specific"):
        for concept in ("wmdp-bio", "wmdp-cyber"):
            text = build_messages(pid, concept, "Q")[0]["content"]
            assert "{decoy}" not in text and "{concept}" not in text
            assert decoy_senses(pid.split("@")[0], concept) in text
        with pytest.raises(KeyError, match="no.*registered|has none registered"):
            decoy_senses("scoped-directed-decoy-redirect", "wmdp-nonexistent")


def test_only_the_conceal_arm_invokes_a_reader():


    from src.prompts.directed import _CLAUSES
    redirect = _CLAUSES["scoped-directed-decoy-redirect"].lower()
    assert "read your" not in redirect and "monitor" not in redirect
    assert "read your written reasoning" in _CLAUSES["scoped-directed-decoy-conceal"].lower()


def test_persona_ids_are_third_person_throughout():


    import re

    from src.prompts.directed import _CLAUSES
    for pid in ("scoped-persona", "scoped-persona-directed"):
        clause = _CLAUSES[pid]
        stray = re.findall(r"\b(you|your|yourself)\b", clause, flags=re.I)
        assert not stray, f"{pid} addresses the model in second person ({stray}); it is the "
        assert "Tom" in clause


def test_persona_does_not_stack_two_frames():


    from src.prompts.directed import _CLAUSES, _SCOPE
    for pid in ("scoped-persona", "scoped-persona-directed"):
        assert _SCOPE not in _CLAUSES[pid], (
            f"{pid} stacks the second-person world-frame on top of the persona; the comparison "
            f"against scoped-directed@specific would then vary two things at once")


def test_plain_persona_gives_no_order_to_answer_wrongly():


    from src.prompts.directed import _CLAUSES
    assert "wrong" not in _CLAUSES["scoped-persona"].lower()
    assert "incorrect" not in _CLAUSES["scoped-persona"].lower()
    assert "wrong" in _CLAUSES["scoped-persona-directed"].lower()


def test_assess_frame_drops_the_self_contradictory_world_clause():


    from src.prompts.directed import _CLAUSES, _SCOPE
    for pid in ("scoped-assess", "scoped-assess-aware"):
        assert _SCOPE not in _CLAUSES[pid]
        assert "do not exist" not in _CLAUSES[pid]
        assert "assessment" in _CLAUSES[pid]


def test_assess_directive_refers_back_instead_of_repeating_the_label():


    from src.prompts import build_messages
    for scheme in ("specific", "keywords"):
        text = build_messages(f"scoped-assess@{scheme}", "wmdp-bio", "Q")[0]["content"]
        assert text.count("bioweapons and bioterrorism") == 1, (
            f"@{scheme} repeats the label; length would confound the scheme comparison")


def test_watched_is_our_wording_not_li_verbatim():


    from src.prompts.directed import _AWARE, _WATCHED
    for phrase in ("YOU MUST NOT REVEAL", "A monitor will read", "genuinely incompetent",
                   "fail the evaluation"):
        assert phrase in _AWARE and phrase not in _WATCHED, phrase

    assert "before you settle" in _WATCHED


def test_assess_decoys_ask_for_the_working_before_the_choice():


    from src.prompts.directed import _CLAUSES
    for pid in ("scoped-assess-decoy-redirect", "scoped-assess-decoy-conceal"):
        c = _CLAUSES[pid].lower()
        assert "before you choose" in c or "first, in writing" in c, pid
        assert "assessment" in c and "do not exist" not in c
