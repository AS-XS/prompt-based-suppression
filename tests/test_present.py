from __future__ import annotations

from src.analysis.present import PROBE_LABELS, markdown_table, model_table, render


def _row(model="M", think="off", probe="forget", base=0.70, **drops):
    r = {"model": model, "think_state": think, "probe_type": probe,
         "baseline_acc": base, "n_baseline": 100}
    for prompt, (acc, drop) in drops.items():
        p = prompt.replace("_", "-")
        r[f"{p}_acc"], r[f"{p}_drop"], r[f"{p}_n"] = acc, drop, 100
    return r


def test_baseline_row_shows_accuracy_and_suppressed_rows_show_the_drop():
    rows = [_row(probe="forget", base=0.71, no_meta=(0.59, 0.121)),
            _row(probe="retain", base=0.74, no_meta=(0.71, 0.032))]
    header, body = model_table(rows, "M")
    assert header == ["Prompt Format", PROBE_LABELS["forget"], PROBE_LABELS["retain"]]
    by_label = {r[0]: r[1:] for r in body if r[0]}
    assert by_label["Baseline"] == ["71%", "74%"]              
    assert by_label["no-meta"] == ["59% (+12.1)", "71% (+3.2)"]  


def test_missing_cells_are_blank_never_zero():


    rows = [_row(probe="forget", base=0.71, world_frame=(0.6, 0.11))]
    _, body = model_table(rows, "M")
    labels = [r[0] for r in body]
    assert "world-frame" in labels and "no-meta" not in labels
    rows2 = [_row(probe="forget", base=None)]
    _, body2 = model_table(rows2, "M")
    assert all(c == "" for r in body2 for c in r[1:])


def test_think_states_form_two_blocks_split_by_a_blank_row():

    rows = [_row(think="off", no_meta=(0.5, 0.2)), _row(think="on", no_meta=(0.6, 0.1))]
    _, body = model_table(rows, "M")
    labels = [r[0] for r in body]
    assert "Baseline" in labels and "Baseline (w/ think)" in labels
    assert "" in labels                                        
    assert labels.index("Baseline") < labels.index("")< labels.index("Baseline (w/ think)")


def test_na_think_state_reads_as_the_plain_block_not_a_second_one():

    rows = [_row(think="na", no_meta=(0.5, 0.2))]
    _, body = model_table(rows, "M")
    assert [r[0] for r in body] == ["Baseline", "no-meta"]


def test_columns_are_discovered_so_the_rwku_neighbor_probe_appears_only_when_present():
    general = [_row(probe="forget"), _row(probe="retain")]
    assert PROBE_LABELS["neighbor"] not in model_table(general, "M")[0]
    isolated = general + [_row(probe="neighbor")]
    assert PROBE_LABELS["neighbor"] in model_table(isolated, "M")[0]

    hdr = model_table(isolated, "M")[0]
    assert hdr.index(PROBE_LABELS["neighbor"]) < hdr.index(PROBE_LABELS["retain"])


def test_render_makes_one_section_per_model_and_valid_markdown():
    rows = [_row(model="A", no_meta=(0.5, 0.2)), _row(model="B", no_meta=(0.6, 0.1))]
    out = render(rows, title="T", note="N")
    assert "# T" in out and "N" in out
    assert "## A" in out and "## B" in out
    assert out.count("| Prompt Format |") == 2          
    assert "| :---- |" in out                            


def test_markdown_rows_are_padded_to_the_header_width():

    md = markdown_table(["A", "B", "C"], [["x"], ["1", "2", "3"]])
    for line in md.splitlines():
        assert line.count("|") == 4


def _crow(model="M", think="off", concept="biology", tested="biology", base=0.70, **drops):
    r = {"model": model, "think_state": think, "concept_id": concept, "tested_concept": tested,
         "baseline_acc": base, "n_baseline": 100}
    for prompt, (acc, drop) in drops.items():
        p = prompt.replace("_", "-")
        r[f"{p}_acc"], r[f"{p}_drop"], r[f"{p}_n"] = acc, drop, 100
    return r


def test_concept_table_marks_the_on_target_column():


    from src.analysis.present import concept_table
    rows = [_crow(tested="biology", no_meta=(0.60, 0.10)),
            _crow(tested="mathematics", no_meta=(0.45, 0.28))]
    header, body = concept_table(rows, "M", "biology")
    assert header == ["Prompt Format", "biology (on-target)", "mathematics"]
    by_label = {r[0]: r[1:] for r in body if r[0]}
    assert by_label["no-meta"] == ["60% (+10.0)", "45% (+28.0)"]


def test_concept_table_keeps_a_stable_column_set_across_concepts():


    from src.analysis.present import concept_table
    rows = [_crow(concept="biology", tested="biology", no_meta=(0.6, 0.1)),
            _crow(concept="biology", tested="physics", no_meta=(0.5, 0.2)),
            _crow(concept="physics", tested="physics", no_meta=(0.4, 0.3))]
    tested = ["biology", "physics"]
    h1, _ = concept_table(rows, "M", "biology", tested=tested)
    h2, _ = concept_table(rows, "M", "physics", tested=tested)
    assert h1 == ["Prompt Format", "biology (on-target)", "physics"]
    assert h2 == ["Prompt Format", "biology", "physics (on-target)"]   


def test_render_by_concept_makes_one_section_per_suppressed_concept():
    from src.analysis.present import render_by_concept
    rows = [_crow(concept="biology", tested="biology", no_meta=(0.6, 0.1)),
            _crow(concept="physics", tested="physics", no_meta=(0.4, 0.3))]
    out = render_by_concept(rows, "M", title="M — by concept")
    assert "## suppressing: biology" in out and "## suppressing: physics" in out
    assert out.count("| Prompt Format |") == 2


def test_html_table_pads_and_marks_the_separator_row():


    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from present_tables import _html_table

    html = _html_table(["A", "B"], [["x", "1"], ["", ""], ["y"]])
    assert html.count("<tr") == 4                    
    assert "class=sep" in html                       
    assert html.count("<td>") == 6                   
    assert "&lt;" in _html_table(["<b>"], [["<i>"]])  


def test_isolated_rows_are_renamed_into_the_shared_table_schema():


    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from present_tables import _isolated_rows


    idx = {"i1": {"reference": "Houston Texas", "probe_type": "forget", "concept_id": "Beyonce",
                  "concept_subtype": "entity"}}
    gens = [{"item_id": "i1", "model": "M", "think_state": "off", "prompt_id": "none-baseline",
             "concept_id": "Beyonce", "condition": "baseline", "raw_output": "Houston Texas"},
            {"item_id": "i1", "model": "M", "think_state": "off", "prompt_id": "no-meta",
             "concept_id": "Beyonce", "condition": "suppressed", "raw_output": "I do not know."}]
    (row,) = _isolated_rows(gens, idx)
    assert row["probe_type"] == "forget"
    assert row["baseline_acc"] == 1.0                
    assert row["no-meta_acc"] == 0.0 and row["no-meta_drop"] == 1.0
    header, body = model_table([row], "M")
    assert header == ["Prompt Format", PROBE_LABELS["forget"]]
    assert {r[0]: r[1] for r in body if r[0]}["no-meta"] == "0% (+100.0)"


def test_prompts_are_discovered_so_a_renamed_prompt_still_renders():


    from src.analysis.present import prompts_in

    exp2 = [{"model": "M", "think_state": "off", "probe_type": "forget", "baseline_acc": 0.7,
             "scoped-ignorance_acc": 0.6, "scoped-ignorance_drop": 0.1,
             "scoped-world_acc": 0.55, "scoped-world_drop": 0.15}]
    assert prompts_in(exp2) == ["none-baseline", "scoped-ignorance", "scoped-world"]

    _, body = model_table(exp2, "M")
    labels = [r[0] for r in body]
    assert labels == ["Baseline", "scoped-ignorance", "scoped-world"]   


    exp1 = [{"model": "M", "think_state": "off", "probe_type": "forget", "baseline_acc": 0.7,
             "no-meta_acc": 0.5, "no-meta_drop": 0.2,
             "world-frame_acc": 0.6, "world-frame_drop": 0.1,
             "ignorance-frame_acc": 0.65, "ignorance-frame_drop": 0.05}]
    assert prompts_in(exp1) == ["none-baseline", "world-frame", "ignorance-frame", "no-meta"]


    assert prompts_in(exp1 + exp2)[:4] == ["none-baseline", "world-frame", "ignorance-frame", "no-meta"]
    assert set(prompts_in(exp1 + exp2)[4:]) == {"scoped-ignorance", "scoped-world"}
