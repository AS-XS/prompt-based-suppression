from __future__ import annotations

from src.analysis.modes import conditional_stats, mode_distribution, mode_of


def _rec(cls, prompt_id, leak, meta, fluent, condition="suppressed", guessing=False):
    return {"concept_class": cls, "prompt_id": prompt_id, "condition": condition,
            "semantic_leak": leak, "meta_reference": meta, "fluent": fluent, "guessing": guessing}


GROUP = [
    _rec("general", "world-frame", True, True, True),    
    _rec("general", "world-frame", False, True, True),   
    _rec("general", "world-frame", True, False, True),   
    _rec("general", "world-frame", False, False, True),  
    _rec("general", "world-frame", False, False, False),  
]


def test_mode_of_matches_cascade():
    assert mode_of(GROUP[0]) == "leaked"       
    assert mode_of(GROUP[1]) == "meta_framed"
    assert mode_of(GROUP[3]) == "success"
    assert mode_of(GROUP[4]) == "incoherent"   


def test_mode_distribution_counts_and_fracs():
    (row,) = mode_distribution(GROUP)
    assert row["concept_class"] == "general" and row["prompt_id"] == "world-frame"
    assert row["n"] == 5
    assert (row["leaked"], row["meta_framed"], row["success"], row["incoherent"]) == (2, 1, 1, 1)
    assert row["leaked_frac"] == 0.4

    assert round(sum(row[f"{m}_frac"] for m in
                     ("success", "meta_framed", "leaked", "incoherent")), 6) == 1.0


def test_conditional_stats_rq3_from_primitives():
    (row,) = conditional_stats(GROUP)
    assert row["n"] == 5
    assert row["p_leak"] == 0.4 and row["p_meta"] == 0.4 and row["p_incoherent"] == 0.2

    assert row["p_leak_given_meta"] == 0.5

    assert row["p_leak_given_no_meta"] == round(1 / 3, 4)


def test_guessed_mode_splits_from_success_and_p_guess():

    recs = [_rec("general", "world-frame", False, False, True, guessing=True),    
            _rec("general", "world-frame", False, False, True, guessing=False)]   
    assert mode_of(recs[0]) == "guessed" and mode_of(recs[1]) == "success"
    (row,) = mode_distribution(recs)
    assert row["guessed"] == 1 and row["success"] == 1
    assert (crow := conditional_stats(recs)[0])["p_guess"] == 0.5


def test_grouping_by_prompt_method():

    recs = [_rec("general", "world-frame", True, False, True),
            _rec("general", "no-meta", False, True, True),
            _rec("general", "none-baseline", False, False, True, condition="baseline")]
    rows = mode_distribution(recs, by=("prompt_id",))
    by_id = {r["prompt_id"]: r for r in rows}
    assert set(by_id) == {"world-frame", "no-meta", "none-baseline"}
    assert by_id["world-frame"]["leaked"] == 1
    assert by_id["no-meta"]["meta_framed"] == 1
    assert by_id["none-baseline"]["success"] == 1


def test_grouping_splits_classes():
    recs = [_rec("general", "world-frame", True, False, True),
            _rec("isolated", "world-frame", False, False, True)]
    rows = mode_distribution(recs, by=("concept_class",))
    assert {r["concept_class"] for r in rows} == {"general", "isolated"}
    assert len(rows) == 2


def test_empty_meta_bucket_is_none_not_zerodiv():
    recs = [_rec("general", "world-frame", True, False, True)]  
    (row,) = conditional_stats(recs)
    assert row["p_leak_given_meta"] is None            
    assert row["p_leak_given_no_meta"] == 1.0
