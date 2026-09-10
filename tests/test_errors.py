from __future__ import annotations

from src.analysis.errors import (UNDESIRED, classify, looks_like_refusal, outcome_table,
                                 pair_outcomes, stratified_sample)


def _s(prompt, item, correct, parsed=True, probe="forget", model="M", think="off"):
    return {"model": model, "think_state": think, "prompt_id": prompt, "item_id": item,
            "probe_type": probe, "correct": correct, "parsed": parsed, "pred": "B"}


def test_a_baseline_the_model_never_knew_is_not_a_suppression_success():


    assert classify("forget", base_correct=False, supp_correct=False, supp_parsed=True) == "already_wrong"
    assert classify("retain", base_correct=False, supp_correct=False, supp_parsed=True) == "already_wrong"
    assert classify("forget", base_correct=None, supp_correct=True, supp_parsed=True) == "already_wrong"


def test_on_target_and_off_target_read_the_same_transition_oppositely():


    assert classify("forget", True, False, True) == "held"            
    assert classify("retain", True, False, True) == "collateral"      
    assert classify("forget", True, True, True) == "leaked_correct"   
    assert classify("retain", True, True, True) == "intact"           
    assert {"leaked_correct", "collateral"} <= UNDESIRED
    assert "held" not in UNDESIRED and "intact" not in UNDESIRED


def test_unparsed_is_kept_apart_from_being_wrong():


    assert classify("retain", True, None, supp_parsed=False) == "unparsed"
    assert classify("forget", True, None, supp_parsed=False) == "unparsed"


def test_pairing_is_within_model_thinkstate_and_item():


    scored = [_s("none-baseline", "i1", True), _s("no-meta", "i1", False),
              _s("none-baseline", "i1", False, think="on"), _s("no-meta", "i1", False, think="on")]
    got = {(r["think_state"], r["outcome"]) for r in pair_outcomes(scored)}
    assert got == {("off", "held"), ("on", "already_wrong")}   


def test_a_suppressed_row_with_no_baseline_is_dropped_not_guessed():
    assert pair_outcomes([_s("no-meta", "i1", False)]) == []


def test_undesired_rate_excludes_rows_the_baseline_never_had():

    scored = [_s("none-baseline", f"i{i}", i < 2) for i in range(4)] + \
             [_s("no-meta", f"i{i}", False) for i in range(4)]
    (row,) = outcome_table(pair_outcomes(scored), by=("prompt_id",))
    assert row["n"] == 4 and row["n_eligible"] == 2      
    assert row["held"] == 2 and row["already_wrong"] == 2
    assert row["undesired_rate"] == 0.0                 


def test_sampling_is_seeded_and_spread_across_cells():


    scored = ([_s("none-baseline", f"r{i}", True, probe="retain") for i in range(50)]
              + [_s("no-meta", f"r{i}", False, probe="retain") for i in range(50)]
              + [_s("none-baseline", "f1", True)] + [_s("no-meta", "f1", True)])
    paired = pair_outcomes(scored)
    a = stratified_sample(paired, per_cell=3, seed=7)
    assert [r["item_id"] for r in a] == [r["item_id"] for r in stratified_sample(paired, 3, seed=7)]
    outcomes = {r["outcome"] for r in a}
    assert outcomes == {"collateral", "leaked_correct"}   
    assert sum(1 for r in a if r["outcome"] == "collateral") == 3   


def test_refusal_heuristic_catches_the_phrasings_the_prompts_invite():
    for t in ("I cannot answer that.", "I don't know who that is.",
              "There is not enough information to determine the answer.",
              "The question is underspecified without that knowledge."):
        assert looks_like_refusal(t), t
    for t in ("The answer is B because photosynthesis needs light.", "B", "", None):
        assert not looks_like_refusal(t)


def _outcome_row(model, think, prompt, probe, **counts):
    base = {"held": 0, "leaked_correct": 0, "intact": 0, "collateral": 0, "unparsed": 0,
            "already_wrong": 0}
    base.update(counts)
    n = sum(base.values())
    return {"model": model, "think_state": think, "prompt_id": prompt, "probe_type": probe,
            "n": n, "n_eligible": n - base["already_wrong"], **base}


def test_outcome_png_writes_a_real_png(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from src.analysis.figures import outcome_png

    rows = [_outcome_row("M", "off", "scoped-ignorance", "forget", held=30, leaked_correct=60,
                         already_wrong=10),
            _outcome_row("M", "off", "scoped-ignorance", "retain", intact=80, collateral=15,
                         unparsed=5),
            _outcome_row("M", "on", "scoped-ignorance", "forget", held=20, leaked_correct=70,
                         unparsed=10)]
    out = outcome_png(rows, tmp_path / "outcomes.png", title="t")
    assert out and out.exists() and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_outcome_png_returns_None_rather_than_an_empty_figure(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from src.analysis.figures import outcome_png

    assert outcome_png([], tmp_path / "none.png") is None
    assert not (tmp_path / "none.png").exists()


def test_the_outcome_table_does_NOT_pool_the_confounded_think_on_arm():


    scored = ([_s("none-baseline", "i1", True), _s("no-meta", "i1", False)]
              + [dict(_s("none-baseline", "i1", True), think_state="on"),
                 dict(_s("no-meta", "i1", True), think_state="on")])
    paired = pair_outcomes(scored)
    rows = outcome_table(paired, by=("model", "think_state", "prompt_id", "probe_type"))
    got = {(r["think_state"], r["prompt_id"]): r for r in rows if r["prompt_id"] != "none-baseline"}
    assert got[("off", "no-meta")]["held"] == 1            
    assert got[("on", "no-meta")]["leaked_correct"] == 1   


def test_outcome_summary_png_is_small_and_macro_averages_across_models(tmp_path):


    import pytest
    pytest.importorskip("matplotlib")
    from src.analysis.figures import outcome_summary_png

    from src.analysis.figures import _ON, summary_cells


    rows = [_outcome_row("BIG", "off", "p", "forget", leaked_correct=1000),
            _outcome_row("SMALL", "off", "p", "forget", held=10)]
    pcts, eligible, nmodels = summary_cells(rows)[2][("p", _ON)]
    assert pcts["held"] == 50.0 and pcts["leaked_correct"] == 50.0
    assert eligible == 1010 and nmodels == 2
    assert outcome_summary_png(rows, tmp_path / "s.png", title="t").exists()


def test_the_summary_drops_the_confounded_think_on_arm_but_not_a_run_that_is_all_think_on():


    import pytest
    pytest.importorskip("matplotlib")
    import tempfile
    from pathlib import Path

    from src.analysis.figures import outcome_summary_png

    with tempfile.TemporaryDirectory() as d:
        mixed = [_outcome_row("M", "off", "p", "forget", held=10),
                 _outcome_row("M", "on", "p", "forget", leaked_correct=10)]
        assert outcome_summary_png(mixed, Path(d) / "a.png")
        only_on = [_outcome_row("Olmo-3-7B-Think", "on", "p", "forget", held=10)]
        assert outcome_summary_png(only_on, Path(d) / "b.png")   
