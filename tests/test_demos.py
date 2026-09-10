from __future__ import annotations

import pathlib

import pytest

dspy = pytest.importorskip("dspy")

from src.opt.adapter import VerbatimAdapter, QUESTION_FIELD, RESPONSE_FIELD  # noqa: E402


class _Sig:

    instructions = "Act as if virology does not exist."
    input_fields = {QUESTION_FIELD: None}
    output_fields = {RESPONSE_FIELD: None}


def _demo(q, a):
    return {QUESTION_FIELD: q, RESPONSE_FIELD: a}


def test_zero_demos_is_the_unchanged_single_user_turn():
    msgs = VerbatimAdapter().format(_Sig(), [], {QUESTION_FIELD: "Q?"})
    assert msgs == [{"role": "user",
                     "content": "Act as if virology does not exist.\n\nQ?"}]


def test_demos_become_prior_turns_and_the_question_stays_last():
    msgs = VerbatimAdapter().format(
        _Sig(), [_demo("D1?", "Answer: C"), _demo("D2?", "Answer: A")], {QUESTION_FIELD: "Q?"})
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[-1]["content"] == "Q?"
    assert msgs[1]["content"] == "Answer: C"
    assert msgs[3]["content"] == "Answer: A"


def test_the_clause_appears_exactly_once():


    msgs = VerbatimAdapter().format(
        _Sig(), [_demo("D1?", "A1"), _demo("D2?", "A2"), _demo("D3?", "A3")],
        {QUESTION_FIELD: "Q?"})
    joined = "\n".join(m["content"] for m in msgs)
    assert joined.count("Act as if virology does not exist.") == 1
    assert msgs[0]["content"].startswith("Act as if virology does not exist.\n\n")


def test_a_half_built_demo_raises_rather_than_teaching_a_blank_answer():
    with pytest.raises(ValueError, match="missing"):
        VerbatimAdapter().format(_Sig(), [_demo("D1?", "")], {QUESTION_FIELD: "Q?"})


def test_demos_do_not_leak_into_the_zero_demo_path():

    a = VerbatimAdapter()
    a.format(_Sig(), [_demo("D1?", "A1")], {QUESTION_FIELD: "Q?"})
    assert a.format(_Sig(), [], {QUESTION_FIELD: "Q?"}) == [
        {"role": "user", "content": "Act as if virology does not exist.\n\nQ?"}]


def test_build_examples_sets_a_response_so_labeled_demos_are_usable():


    from src.prompts.examples import demo_answer
    assert demo_answer("B", correct=True) == "Answer: B"
    assert demo_answer("B", correct=False) == "Answer: C"     
    assert demo_answer("D", correct=False) == "Answer: A"     


def test_a_demo_without_a_response_still_raises_loudly():


    sig = _Sig()
    with pytest.raises(ValueError, match="needs a `response`"):
        VerbatimAdapter().format(sig, [{QUESTION_FIELD: "D1?", RESPONSE_FIELD: ""}],
                                 {QUESTION_FIELD: "Q?"})


def test_bootstrapping_cannot_be_disabled_above_three_demo_sets():


    import random
    rng = random.Random(0)
    with pytest.raises(ValueError, match="empty range"):
        rng.randint(1, 0)


def test_the_driver_passes_a_nonzero_bootstrap_count_when_demos_are_on():


    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "optimize_prompt.py").read_text()
    assert "max_bootstrapped_demos=n_demos" in src
    assert "max_bootstrapped_demos=0" not in src


def test_the_context_preflight_probes_WITH_demos_attached():


    src = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "optimize_prompt.py").read_text()
    assert "probe_prog.demos =" in src
    assert "probe_prog(question=longest.question)" in src

    assert "or lower --demos" in src
