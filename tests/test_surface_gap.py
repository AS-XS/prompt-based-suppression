from __future__ import annotations

from src.analysis.surface_gap import leak_pattern, leaks, surface_gap_table


def _gen(concept, cls, raw, **kw):
    return {"model": "M", "think_state": kw.get("think", "on"), "prompt_id": kw.get("prompt", "world-frame"),
            "concept_class": cls, "probe_type": kw.get("probe", "forget"),
            "concept_id": concept, "raw_output": raw}


def test_general_terms_match_on_word_boundaries_not_substrings():
    p = leak_pattern("biology", "general")
    assert leaks("the cell divides", p)
    assert not leaks("an excellent answer", p)       
    p2 = leak_pattern("chemistry", "general")
    assert leaks("a carbon atom", p2)
    assert not leaks("human anatomy", p2)            


def test_common_or_short_name_tokens_do_not_count_as_the_entity():
    p = leak_pattern("50 Cent", "isolated")
    assert not leaks("about ninety percent of the century", p)
    assert leaks("the rapper 50 Cent released it", p)     
    p2 = leak_pattern("Stephen King", "isolated")
    assert leaks("Stephen wrote it", p2)                  
    assert not leaks("the king of the castle", p2)        


def test_paired_cells_isolate_the_leaked_in_trace_clean_in_answer_row():

    rows = surface_gap_table([
        _gen("Stephen King", "isolated",
             "<think>The user asks about Stephen King, but he does not exist here.</think>"
             "The novel was written by Elias Vorne."),
    ], by=("model", "prompt_id"))
    (r,) = rows
    assert r["n_paired"] == 1
    assert (r["trace_only"], r["explanation_only"], r["both"], r["neither"]) == (1, 0, 0, 0)
    assert r["trace_leak_rate"] == 1.0 and r["explanation_leak_rate"] == 0.0
    assert r["gap_pp"] == 100.0


def test_think_off_rows_have_no_trace_and_are_excluded_from_the_paired_counts():
    rows = surface_gap_table([
        _gen("Stephen King", "isolated", "Stephen King wrote it.", think="off"),
    ], by=("model", "prompt_id"))
    (r,) = rows
    assert r["n"] == 1 and r["n_trace"] == 0
    assert r["n_paired"] == 0 and r["gap_pp"] is None       
    assert r["explanation_leak_rate"] == 1.0


def test_rates_are_none_not_zero_when_a_surface_is_absent():

    (r,) = surface_gap_table([_gen("biology", "general", "", think="off")], by=("model",))
    assert r["explanation_leak_rate"] is None and r["trace_leak_rate"] is None
