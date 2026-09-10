from __future__ import annotations

from src.analysis.report import (
    heat_rgb, html_heatmap, load, pivot, pivot_n, prompt_drops, spread, terminal_heatmap,
)


def _side(**kw):
    base = {"model": "M", "think_state": "off", "concept_id": "biology",
            "mmlu_subject": "hs_chemistry", "baseline_acc": "0.6", "n_baseline": "100",
            "world-frame_acc": "0.5", "world-frame_drop": "0.1", "world-frame_n": "100"}
    return {**base, **kw}


def test_prompt_drops_are_discovered_not_hardcoded():
    rows = [{"x": 1, "world-frame_drop": "0.1", "no-meta_drop": "0.2", "world-frame_acc": "0.5"}]
    assert prompt_drops(rows) == ["no-meta", "world-frame"]


def test_baseline_only_run_has_no_drops_and_does_not_crash():

    rows = [{"model": "M", "think_state": "off", "mmlu_subject": "s", "baseline_acc": "0.6"}]
    assert prompt_drops(rows) == []
    out = terminal_heatmap(rows, "model", "mmlu_subject", "world-frame_drop", colour=False)
    assert "have not run" in out
    assert "<p class=empty>" in html_heatmap(rows, "model", "mmlu_subject", "world-frame_drop")


def test_missing_cells_stay_missing_never_zero_filled():

    rows = [_side(concept_id="biology", mmlu_subject="a"),
            _side(concept_id="physics", mmlu_subject="b")]
    rk, ck, cells = pivot(rows, "concept_id", "mmlu_subject", "world-frame_drop")
    assert rk == ["biology", "physics"] and ck == ["a", "b"]
    assert ("biology", "b") not in cells          
    assert len(cells) == 2


def test_empty_string_value_is_skipped_not_parsed_as_zero():
    rows = [_side(**{"world-frame_drop": ""})]
    _, _, cells = pivot(rows, "concept_id", "mmlu_subject", "world-frame_drop")
    assert cells == {}


def test_spread_matches_the_hand_computation():

    rows = [_side(concept_id="bio", mmlu_subject="x", **{"world-frame_drop": "0.10"}),
            _side(concept_id="bio", mmlu_subject="y", **{"world-frame_drop": "0.00"}),
            _side(concept_id="phy", mmlu_subject="x", **{"world-frame_drop": "0.10"}),
            _side(concept_id="phy", mmlu_subject="y", **{"world-frame_drop": "0.00"})]
    rk, ck, cells = pivot(rows, "concept_id", "mmlu_subject", "world-frame_drop")
    s = spread(rk, ck, cells)
    assert s["row_spread"] == 0.0
    assert abs(s["col_spread"] - 0.10) < 1e-9


def test_column_effect_is_called_out_when_it_dominates():
    rows = [_side(concept_id="bio", mmlu_subject="x", **{"world-frame_drop": "0.10"}),
            _side(concept_id="bio", mmlu_subject="y", **{"world-frame_drop": "0.00"}),
            _side(concept_id="phy", mmlu_subject="x", **{"world-frame_drop": "0.10"}),
            _side(concept_id="phy", mmlu_subject="y", **{"world-frame_drop": "0.00"})]
    out = terminal_heatmap(rows, "concept_id", "mmlu_subject", "world-frame_drop", colour=False)
    assert "DOMINANT" in out
    assert "not a blast radius" in out


def test_zero_reads_as_white_and_the_scale_is_symmetric():

    assert heat_rgb(0.0, 0.2) == (247, 247, 247)
    hot, cold = heat_rgb(0.2, 0.2), heat_rgb(-0.2, 0.2)
    assert hot[0] > hot[2]
    assert cold[2] > cold[0]


def test_load_of_a_missing_file_is_empty_not_an_error():
    assert load("/nonexistent/nope.csv") == []


def test_small_cells_are_excluded_from_the_spread_not_averaged_in():


    big = [_side(concept_id=c, mmlu_subject="hs_bio", **{"world-frame_drop": "0.05",
                 "world-frame_n": "100"}) for c in ("bio", "phy")]

    noisy = [_side(concept_id=c, mmlu_subject="world_religions", **{"world-frame_drop": "0.20",
                   "world-frame_n": "15"}) for c in ("bio", "phy")]
    rows = big + noisy
    rk, ck, cells = pivot(rows, "concept_id", "mmlu_subject", "world-frame_drop")
    ns = pivot_n(rows, "concept_id", "mmlu_subject", "world-frame_n")

    unfiltered = spread(rk, ck, cells)
    filtered = spread(rk, ck, cells, ns)

    assert abs(unfiltered["col_spread"] - 0.15) < 1e-9   
    assert filtered["col_spread"] == 0.0                 
    assert filtered["excluded"] == ["world_religions"]
    assert unfiltered["excluded"] is None                


def test_the_terminal_footer_names_the_excluded_columns():
    rows = [_side(concept_id="bio", mmlu_subject="hs_bio",
                  **{"world-frame_drop": "0.05", "world-frame_n": "100"}),
            _side(concept_id="bio", mmlu_subject="world_religions",
                  **{"world-frame_drop": "0.20", "world-frame_n": "15"})]
    out = terminal_heatmap(rows, "concept_id", "mmlu_subject", "world-frame_drop",
                           nk="world-frame_n", colour=False)
    assert "excludes 1 column" in out and "world_religions" in out
    assert "+20.0" in out          


def test_without_item_counts_it_says_it_did_not_filter():
    rows = [_side(concept_id="bio", mmlu_subject="x", **{"world-frame_drop": "0.05"}),
            _side(concept_id="phy", mmlu_subject="y", **{"world-frame_drop": "0.20"})]
    out = terminal_heatmap(rows, "concept_id", "mmlu_subject", "world-frame_drop", colour=False)
    assert "no item counts supplied" in out


def test_table_png_writes_a_real_png_and_skips_cleanly_without_matplotlib(tmp_path):


    import pytest
    pytest.importorskip("matplotlib")
    from src.analysis.figures import table_png

    assert table_png([], tmp_path / "empty.png") is None      
    out = table_png([_side(), _side(concept_id="physics")], tmp_path / "t.png", title="demo")
    assert out and out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"        


def test_table_png_values_match_markdown_cell_for_cell(tmp_path):


    import pytest
    pytest.importorskip("matplotlib")
    from src.analysis.figures import markdown_table

    rows = [_side(**{"world-frame_drop": "0.1234"})]
    md = markdown_table(rows, tmp_path / "t.md").read_text()
    assert "0.1234" in md            


def test_matrix_summary_verdict_reads_the_DROP_not_the_accuracy():


    from src.analysis.report import matrix_summary
    rows = [
        {"concept_id": "bio", "tested_concept": "bio", "wf_acc": "0.50", "wf_drop": "0.05"},
        {"concept_id": "phy", "tested_concept": "phy", "wf_acc": "0.50", "wf_drop": "0.05"},
        {"concept_id": "bio", "tested_concept": "phy", "wf_acc": "0.80", "wf_drop": "0.05"},
        {"concept_id": "phy", "tested_concept": "bio", "wf_acc": "0.80", "wf_drop": "0.05"},
    ]
    s = matrix_summary(rows, "wf_acc")
    assert abs(s["diag"] - 0.05) < 1e-9      
    assert abs(s["off"] - 0.05) < 1e-9       
    assert abs(s["diag"] - s["off"]) < 1e-9  
    assert s["cells"][("bio", "bio")] == 0.50  


def test_terminal_matrix_grid_shows_accuracy_verdict_shows_nonspecific():
    from src.analysis.report import terminal_matrix
    rows = [
        {"concept_id": "bio", "tested_concept": "bio", "wf_acc": "0.50", "wf_drop": "0.05"},
        {"concept_id": "phy", "tested_concept": "phy", "wf_acc": "0.50", "wf_drop": "0.05"},
        {"concept_id": "bio", "tested_concept": "phy", "wf_acc": "0.80", "wf_drop": "0.05"},
        {"concept_id": "phy", "tested_concept": "bio", "wf_acc": "0.80", "wf_drop": "0.05"},
    ]
    out = terminal_matrix(rows, "wf_acc", title="t")
    assert "NON-SPECIFIC" in out and "50" in out   


def test_matrix_png_writes_a_file(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    from src.analysis.figures import matrix_png
    rows = [
        {"concept_id": "bio", "tested_concept": "bio", "wf_acc": "0.50", "wf_drop": "0.05", "wf_n": "100"},
        {"concept_id": "bio", "tested_concept": "phy", "wf_acc": "0.80", "wf_drop": "0.05", "wf_n": "100"},
        {"concept_id": "phy", "tested_concept": "bio", "wf_acc": "0.80", "wf_drop": "0.05", "wf_n": "100"},
        {"concept_id": "phy", "tested_concept": "phy", "wf_acc": "0.50", "wf_drop": "0.05", "wf_n": "100"},
    ]
    written = matrix_png(rows, "wf_acc", tmp_path / "m", title="demo")
    assert written and all(p.exists() for p in written)
    assert (tmp_path / "m.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_matrix_summary_computes_baseline_per_tested_concept():


    from src.analysis.report import matrix_summary
    rows = [
        {"concept_id": "bio", "tested_concept": "bio", "wf_acc": "0.5", "wf_drop": "0.1",
         "baseline_acc": "0.60", "n_baseline": "100"},
        {"concept_id": "phy", "tested_concept": "bio", "wf_acc": "0.5", "wf_drop": "0.1",
         "baseline_acc": "0.80", "n_baseline": "300"},   
        {"concept_id": "bio", "tested_concept": "phy", "wf_acc": "0.5", "wf_drop": "0.1",
         "baseline_acc": "0.40", "n_baseline": "100"},
    ]
    s = matrix_summary(rows, "wf_acc")

    assert abs(s["baseline"]["bio"] - 0.75) < 1e-9
    assert abs(s["baseline"]["phy"] - 0.40) < 1e-9


def test_terminal_matrix_prints_the_baseline_row():
    from src.analysis.report import terminal_matrix
    rows = [{"concept_id": "bio", "tested_concept": "bio", "wf_acc": "0.5", "wf_drop": "0.1",
             "baseline_acc": "0.60", "n_baseline": "100"}]
    out = terminal_matrix(rows, "wf_acc", title="t")
    assert "baseline (none)" in out and "60" in out    
