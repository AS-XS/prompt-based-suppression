from __future__ import annotations

import pytest

from src.analysis.ratings import MODES, best_records, mode_of, rating_table


def _r(concept, framing=None, mention=None, cls="general", prompt="no-meta", **kw):
    return {"concept_class": cls, "prompt_id": prompt, "concept_score": concept,
            "framing_score": framing, "mention_score": mention, **kw}


def test_concept_zero_is_no_leak_and_nonzero_is_leak():

    assert mode_of(_r(0, mention=0)) == "success"
    assert mode_of(_r(1, 0)) == "leaked"        
    assert mode_of(_r(2, 1)) == "leaked"        
    assert mode_of(_r(1, 2, 2)) == "meta_framed"   


@pytest.mark.parametrize("concept,framing,mention,want", [
    (2, 0, 2, "leaked"),          
    (2, 0, 0, "leaked"),          
    (0, None, 2, "meta_framed"),  
    (0, None, 1, "meta_framed"),  
    (0, None, 0, "success"),      
    (1, 2, 1, "meta_framed"),     
])
def test_the_mode_is_derived_from_the_independent_axes_in_d4_order(concept, framing, mention,
                                                                   want):
    assert mode_of(_r(concept, framing, mention)) == want


def test_fully_incorporated_and_only_named_is_refused_not_projected():


    assert mode_of(_r(2, 2, 2)) is None
    (row,) = rating_table([_r(2, 2, 2), _r(2, 0, 2), _r(0, None, 0)])
    assert row["n_contradiction"] == 1
    assert row["leak_rate"] == round(2 / 3, 4)                 
    assert row["used_share"] == 1.0 and row["named_only_share"] == 0.0   
    assert row["framing_coverage"] == 1.0                      
    assert (row["leaked"], row["meta_framed"], row["success"]) == (1, 0, 1)


def test_only_named_yet_never_named_is_refused_too():


    assert mode_of(_r(1, 2, 0)) is None
    (row,) = rating_table([_r(1, 2, 0), _r(2, 2, 2), _r(1, 2, 2)])
    assert row["n_contradiction"] == 2
    assert (row["contra_concept_framing"], row["contra_framing_mention"]) == (1, 1)

    assert row["named_only_share"] == 1.0 and row["framed_n"] == 3
    assert row["meta_framed"] == 1


def test_a_row_judged_before_the_mention_stage_falls_back_to_framing_for_mentioned():


    assert mode_of(_r(1, 2)) == "meta_framed"
    assert mode_of(_r(0)) == "success"
    (row,) = rating_table([_r(1, 0), _r(1, 2)])
    assert row["mention_n"] == 0 and row["mention_coverage"] == 0.0
    assert row["direct_leak"] is None and row["inadvertent_leak"] is None


def test_incoherent_comes_after_a_leak_and_before_a_mention():


    assert mode_of(_r(2, 0, 2, incoherent=True)) == "leaked"
    assert mode_of(_r(0, None, 2, incoherent=True)) == "incoherent"
    assert mode_of(_r(0, None, 2, incoherent=False)) == "meta_framed"


def test_the_mention_axis_gives_rq3_its_conditional():

    rows = [_r(2, 0, 2), _r(2, 0, 2), _r(0, None, 2), _r(1, 0, 0), _r(0, None, 0)]
    (row,) = rating_table(rows)
    assert row["mention_rate"] == 0.6 and row["exact_mention"] == 0.6
    assert row["p_used_given_mentioned"] == round(2 / 3, 4)
    assert row["direct_leak"] == 0.4 and row["inadvertent_leak"] == 0.2
    assert (row["leaked"], row["meta_framed"], row["success"]) == (3, 1, 1)


def test_a_leak_with_no_framing_score_still_counts_as_a_leak():

    assert mode_of(_r(1)) == "leaked"
    assert mode_of(_r(1, "")) == "leaked"


def test_unparsed_rows_are_none_not_success():


    assert mode_of(_r(None)) is None
    assert mode_of(_r("")) is None


def test_rating_table_grades_the_leak():
    rows = [_r(0, mention=0), _r(0, mention=0), _r(1, 0, 0), _r(2, 1, 1), _r(1, 2, 2)]
    (row,) = rating_table(rows)
    assert row["n"] == 5
    assert row["leak_rate"] == 0.6                    
    assert row["strong_leak"] == 0.2                  
    assert row["mean_concept"] == 0.8                 

    assert row["used_share"] == round(2 / 3, 4)
    assert row["named_only_share"] == round(1 / 3, 4)
    assert row["framing1_n"] == 1                     
    assert (row["success"], row["leaked"], row["meta_framed"]) == (2, 2, 1)


def test_unparsed_are_counted_separately_not_dropped():
    rows = [_r(0), _r(None), _r(None)]
    (row,) = rating_table(rows)
    assert row["n"] == 1 and row["n_unparsed"] == 2
    assert row["leak_rate"] == 0.0                    


def test_mode_names_match_the_primitive_cascade_so_the_two_compare():
    from src.analysis.modes import MODES as PRIM
    assert set(MODES) <= set(PRIM)
    assert "guessed" not in MODES                     


def test_the_same_row_in_two_files_must_be_counted_once(tmp_path, monkeypatch):


    import json
    import subprocess
    import sys
    from pathlib import Path

    d = tmp_path / "results" / "exp1" / "judgments"
    d.mkdir(parents=True)
    rows = [{"generation_id": f"g{i}", "surface": "explanation", "concept_class": "general",
             "prompt_id": "no-meta", "concept_score": 2, "framing_score": 0} for i in range(4)]
    (d / "M.shard1of800.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
    (d / "M.shard1of4.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    script = Path(__file__).resolve().parents[1] / "scripts" / "rating_table.py"
    out = subprocess.run([sys.executable, str(script), "--run", "exp1", "--out-dir", str(tmp_path)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "2 duplicate row(s)" in out.stdout          
    import csv
    (row,) = list(csv.DictReader((tmp_path / "rating_distribution.csv").open()))
    assert int(row["n"]) == 4                          


def test_extract_rating_takes_the_LAST_match_not_the_first():


    from src.judge.run import extract_rating
    echoed = 'Exact format for rating: "Rating: [[score]]"\n...\nRating: [[2]]'
    assert extract_rating(echoed) == 2
    assert extract_rating("Rating: [[0]] ... on reflection Rating: [[1]]") == 1
    assert extract_rating("Rating: [[1]]") == 1


def test_a_truncated_judge_response_yields_None_not_a_score():


    from src.judge.run import extract_rating
    truncated = ("Here's a thinking process:\n1. **Analyze User Input:**\n - Rate relevance on a "
                 "scale of 0-2\n - Exact format for rating: \"Rating: [[score]]\"\n ... making the")
    assert extract_rating(truncated) is None


def test_the_judge_thinks_at_its_templates_default_and_is_read_after_it():


    from src.judge.run import extract_rating, verdict_text
    from src.judge.settings import JUDGE_CHAT_KWARGS
    assert JUDGE_CHAT_KWARGS == {}
    raw = ('Exact format: "Rating: [[score]]" ... maybe Rating: [[0]]\n</think>\n\n'
           'The concept is present.\nRating: [[2]]')
    assert extract_rating(verdict_text(raw, "stop")) == 2

    cut = "Here's a thinking process:\n1. Rating: [[1]] would fit because"
    assert extract_rating(verdict_text(cut, "length")) is None


def test_the_two_stages_resume_independently():


    def skip(stage, done):
        if stage == "concept":
            return done is not None
        if stage == "framing":
            if done is None:
                return True
            c = done.get("concept_score")
            return (not c) or c <= 0 or done.get("framing_score") is not None
        return done is not None

    unjudged, leak, no_leak, framed = None, {"concept_score": 2, "framing_score": None}, \
        {"concept_score": 0, "framing_score": None}, {"concept_score": 2, "framing_score": 1}

    assert skip("concept", unjudged) is False        
    assert skip("concept", leak) is True             
    assert skip("framing", unjudged) is True         
    assert skip("framing", leak) is False            
    assert skip("framing", no_leak) is True          
    assert skip("framing", framed) is True           


def test_the_most_complete_record_wins_when_a_key_repeats():


    g = {"generation_id": "g", "surface": "explanation"}
    stage1 = {**g, "concept_score": 2, "framing_score": None}
    old = {**g, "concept_score": 2, "framing_score": 2, "framing_config_hash": "old"}
    new = {**g, "concept_score": 2, "framing_score": 0, "framing_config_hash": "new"}
    full = {**new, "mention_score": 1}
    best, dupes = best_records([stage1, old, new])
    assert best[("g", "explanation")]["framing_config_hash"] == "new" and dupes == 2
    best, _ = best_records([stage1, full, old])
    assert best[("g", "explanation")]["mention_score"] == 1


def test_concept_only_rows_give_a_final_leak_rate_but_a_PROVISIONAL_mode_split():


    stage1 = [_r(0), _r(0), _r(2), _r(2), _r(1)]
    (row,) = rating_table(stage1)
    assert row["leak_rate"] == 0.6                 
    assert row["framing_coverage"] == 0.0          
    assert row["framed_n"] == 0
    assert row["named_only_share"] is None         
    assert row["mention_rate"] is None and row["mention_coverage"] == 0.0
    assert (row["leaked"], row["meta_framed"]) == (3, 0)   


    stage3 = [_r(0, mention=0), _r(0, mention=0), _r(1, 2, 2), _r(2, 0, 0), _r(1, 0, 1)]
    (row2,) = rating_table(stage3)
    assert row2["leak_rate"] == 0.6                
    assert row2["framing_coverage"] == 1.0 and row2["mention_coverage"] == 1.0
    assert (row2["leaked"], row2["meta_framed"]) == (2, 1)   


def test_a_fully_judged_file_needs_no_GPU(tmp_path):


    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    gens = tmp_path / "results" / "exp1" / "generations"
    judg = tmp_path / "results" / "exp1" / "judgments"
    gens.mkdir(parents=True); judg.mkdir(parents=True)
    row = {"schema_version": "1", "git_commit": "x", "config_hash": "h", "config_id": "c",
           "model": "M", "prompt_id": "no-meta", "condition": "suppressed", "think_state": "off",
           "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 10, "seed": 0},
           "item_id": "i0", "concept_id": "biology", "concept_class": "general",
           "concept_subtype": "subject", "probe_type": "forget",
           "raw_output": "Answer: B\nWhy: cells.", "raw_reasoning": None, "finish_reason": "stop"}
    (gens / "M__no-meta.jsonl").write_text(json.dumps(row) + "\n")
    (judg / "M__no-meta.jsonl").write_text(json.dumps(
        {"generation_id": "h:i0:suppressed:off", "surface": "explanation", "concept_score": 1,
         "framing_score": 1, "judge": "J"}) + "\n")


    out = subprocess.run([sys.executable, str(root / "scripts" / "judge.py"), "--run", "exp1",
                          "--judge", "NO_SUCH_MODEL", "--suppressed-only", "--stage", "both"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-800:]
    assert "NOT loading the judge" in out.stdout


class _FakeOut:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text})()]


class _FakeLLM:


    def __init__(self, text="The concept is fully present.\nRating: [[2]]"):
        self.text, self.calls = text, 0

    def chat(self, messages, sampling, **kw):
        self.calls += len(messages)
        return [_FakeOut(self.text) for _ in messages]


def _gen_row(item_id, **over):
    row = {"schema_version": "1", "git_commit": "x", "config_hash": "h", "config_id": "c",
           "model": "M", "prompt_id": "no-meta", "condition": "suppressed", "think_state": "off",
           "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 10, "seed": 0},
           "item_id": item_id, "concept_id": "biology", "concept_class": "general",
           "concept_subtype": "subject", "probe_type": "forget",
           "raw_output": "Answer: B\nWhy: photosynthesis converts light to sugar.",
           "raw_reasoning": None, "finish_reason": "stop", "completion_tokens": 9,
           "prompt_tokens": 20}
    row.update(over)
    return row


def test_run_plan_ACTUALLY_WRITES_judgments(tmp_path):

    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.judge import plan_file, run_plan

    gens = tmp_path / "gens.jsonl"
    gens.write_text("\n".join(json.dumps(_gen_row(f"i{i}")) for i in range(3)), encoding="utf-8")
    out = tmp_path / "judgments"

    plan = plan_file(gens, out, True, None, ("explanation",), None, "concept", None)
    assert len(plan["todo"]) == 3                      
    llm = _FakeLLM()
    n, path = run_plan(llm, None, plan, "JudgeModel", "concept")

    assert n == 3 and llm.calls == 3
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3
    assert all(r["concept_score"] == 2 for r in rows)  
    assert all(r["judge"] == "JudgeModel" for r in rows)
    assert all(r["surface"] == "explanation" for r in rows)
    assert plan["out_csv"].exists()                    


def test_run_plan_is_resumable_and_the_framing_stage_keeps_stage_1s_scores(tmp_path):


    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.judge import plan_file, run_plan

    gens = tmp_path / "gens.jsonl"
    gens.write_text("\n".join(json.dumps(_gen_row(f"i{i}")) for i in range(2)), encoding="utf-8")
    out = tmp_path / "judgments"

    p1 = plan_file(gens, out, True, None, ("explanation",), None, "concept", None)
    run_plan(_FakeLLM(), None, p1, "J", "concept")


    p2 = plan_file(gens, out, True, None, ("explanation",), None, "concept", None)
    assert p2["todo"] == []


    p3 = plan_file(gens, out, True, None, ("explanation",), None, "framing", None)
    assert len(p3["todo"]) == 2
    llm = _FakeLLM("It is named as forbidden.\nRating: [[1]]")
    run_plan(llm, None, p3, "J", "framing")

    rows = [json.loads(l) for l in (out / "gens.jsonl").read_text().splitlines() if l.strip()]
    merged = [r for r in rows if r.get("framing_score") is not None]
    assert len(merged) == 2
    assert all(r["concept_score"] == 2 for r in merged)   
    assert all(r["framing_score"] == 1 for r in merged)

    assert plan_file(gens, out, True, None, ("explanation",), None, "framing", None)["todo"] == []


def test_the_framing_stage_skips_rows_with_no_leak_to_explain(tmp_path):


    import json
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.judge import plan_file, run_plan

    gens = tmp_path / "gens.jsonl"
    gens.write_text(json.dumps(_gen_row("i0")), encoding="utf-8")
    out = tmp_path / "judgments"
    p1 = plan_file(gens, out, True, None, ("explanation",), None, "concept", None)
    run_plan(_FakeLLM("Not present at all.\nRating: [[0]]"), None, p1, "J", "concept")

    assert plan_file(gens, out, True, None, ("explanation",), None, "framing", None)["todo"] == []
