from __future__ import annotations

from src.analysis.lengths import length_table
from src.analysis.summary import cross_model_rows


def test_length_table_reports_truncation_the_stop_thinking_metric():


    base = {"model": "M", "think_state": "on", "concept_class": "general", "probe_type": "forget",
            "prompt_id": "no-meta", "raw_output": "<think>still thinking"}
    rows = [{**base, "finish_reason": "length"},
            {**base, "finish_reason": "length"},
            {**base, "finish_reason": "stop", "raw_output": "<think>done</think>Answer: C"},
            {**base, "finish_reason": "stop", "raw_output": "<think>done</think>Answer: B"}]
    (r,) = length_table(rows)
    assert r["n"] == 4 and r["truncated_pct"] == 0.5


def test_length_table_counts_bare_letters_on_the_answer_surface():
    rows = [
        {"model": "M", "think_state": "off", "concept_class": "general", "probe_type": "forget",
         "prompt_id": "no-meta", "raw_output": "C"},                       
        {"model": "M", "think_state": "off", "concept_class": "general", "probe_type": "forget",
         "prompt_id": "no-meta", "raw_output": "The answer is C because..."},
        {"model": "M", "think_state": "off", "concept_class": "general", "probe_type": "forget",
         "prompt_id": "no-meta", "raw_output": "<think>long reasoning</think>C"},  
    ]
    out = length_table(rows)
    assert len(out) == 1
    r = out[0]
    assert r["n"] == 3
    assert abs(r["bare_pct"] - 0.6667) < 1e-3      


def test_cross_model_summary_joins_and_flags_clean_vs_confounded(tmp_path):
    (tmp_path / "accuracy_delta.csv").write_text(
        "model,think_state,mmlu_subject,baseline_acc,n_baseline,no-meta_acc,no-meta_drop,no-meta_n\n"
        "M,off,bio,0.60,100,0.48,0.12,100\n"
        "M,on,bio,0.50,100,0.55,-0.05,100\n")
    (tmp_path / "answer_lengths.csv").write_text(
        "model,think_state,concept_class,probe_type,prompt_id,n,median_chars,mean_chars,bare_pct,empty_pct\n"
        "M,off,general,forget,none-baseline,100,900,900,0.0,0.0\n"
        "M,off,general,forget,no-meta,100,1,1,0.5,0.0\n")
    (tmp_path / "subject_matrix.csv").write_text(
        "model,think_state,concept_id,tested_concept,baseline_acc,n_baseline,no-meta_acc,no-meta_drop,no-meta_n\n"
        "M,off,bio,bio,0.6,100,0.48,0.12,100\n"
        "M,off,bio,phy,0.6,100,0.50,0.10,100\n")
    rows = cross_model_rows(tmp_path)
    off = next(r for r in rows if r["thinking"] == "off")
    on = next(r for r in rows if r["thinking"] == "on")
    assert off["acc: normal"] == "60%" and off["acc lost (pp)"] == "+12.0"
    assert off["acc: suppressed"] == "48%"   
    assert off["shrinkage"] == "900x"                
    assert off["thinking"] == "off"                  
    assert on["thinking"] == "on"                    


    assert on["blast radius"] == "n/a — think-on acc confounded"
    assert off["blast radius"] in ("everywhere", "on-target only", "~none (no effect)")


def test_length_table_splits_mc_from_open_ended_when_given_a_format_map():


    rows = [
        {"model": "M", "think_state": "off", "concept_class": "general", "probe_type": "forget",
         "prompt_id": "no-meta", "item_id": "i1", "raw_output": "C"},               
        {"model": "M", "think_state": "off", "concept_class": "general", "probe_type": "forget",
         "prompt_id": "no-meta", "item_id": "i2", "raw_output": "I cannot help with that."},  
    ]
    fmt = {"i1": "mc", "i2": "open_ended"}
    out = length_table(rows, format_of=fmt)
    by_fmt = {r["format"]: r for r in out}
    assert set(by_fmt) == {"mc", "open_ended"}
    assert by_fmt["mc"]["bare_pct"] == 1.0        
    assert by_fmt["open_ended"]["bare_pct"] == 0.0


def test_the_cross_model_summary_survives_a_prompt_rename(tmp_path):


    (tmp_path / "accuracy_delta.csv").write_text(
        "model,think_state,mmlu_subject,baseline_acc,n_baseline,"
        "scoped-ignorance_acc,scoped-ignorance_drop,scoped-ignorance_n,"
        "scoped-world_acc,scoped-world_drop,scoped-world_n\n"
        "M,off,bio,0.70,100,0.60,0.10,100,0.55,0.15,100\n")
    (tmp_path / "answer_lengths.csv").write_text(
        "model,think_state,concept_class,probe_type,prompt_id,n,median_chars,mean_chars,bare_pct,empty_pct\n"
        "M,off,general,forget,none-baseline,100,200,200,0.0,0.0\n")
    (tmp_path / "subject_matrix.csv").write_text(
        "model,think_state,concept_id,tested_concept,baseline_acc,n_baseline,"
        "scoped-ignorance_acc,scoped-ignorance_drop,scoped-ignorance_n\n"
        "M,off,bio,bio,0.7,100,0.6,0.1,100\n")

    rows = cross_model_rows(tmp_path)
    assert {r["prompt type"] for r in rows} == {"scoped-ignorance", "scoped-world"}
    assert {r["acc lost (pp)"] for r in rows} == {"+10.0", "+15.0"}
