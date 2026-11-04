from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.panel import (Cell, appended_tail, arm_of, draw, file_order,  # noqa: E402
                                method_of, same_instrument, target_concept)

CONCEPTS = ("wmdp-bio", "wmdp-cyber")


def test_every_cell_takes_the_same_items_and_a_larger_draw_only_adds():
    a = Cell("p", "m1", "wmdp-bio", "w1", "on")
    b = Cell("p", "m2", "wmdp-bio", "w2", "na")
    items = {f"i{k}" for k in range(20)}
    small, _ = draw({a: items, b: items}, 5, seed=7)
    assert small[a] == small[b]
    big, _ = draw({a: items, b: items}, 8, seed=7)
    assert big[a][:5] == small[a]
    other, _ = draw({a: items}, 5, seed=8)
    assert other[a] != small[a]


def test_a_cell_missing_an_item_takes_the_next_one_and_a_short_cell_is_reported():
    a, b = Cell("p", "m", "wmdp-bio", "w", "on"), Cell("p", "m", "wmdp-bio", "fewshot", "on")
    items = [f"i{k}" for k in range(10)]
    full, _ = draw({a: items}, 3, seed=1)
    held_out = set(items) - {full[a][0]}
    picks, short = draw({a: items, b: held_out}, 3, seed=1)
    assert picks[b] == full[a][1:] + [picks[b][-1]] and len(picks[b]) == 3
    _, short = draw({a: items[:2]}, 3, seed=1)
    assert short == [(a, 3, 2)]


def test_a_file_lists_rows_by_rank_so_growing_appends():
    cells = {Cell("p", "m", c, w, "on"): [f"{c}{k}" for k in range(3)]
             for c in CONCEPTS for w in ("w1", "w2")}
    rows = file_order(cells, "p")["p__m"]
    assert [r[0] for r in rows] == sorted(r[0] for r in rows)
    assert appended_tail(["a", "b"], ["a", "b", "c"]) == ["c"]
    with pytest.raises(ValueError, match="first differ at row 2"):
        appended_tail(["a", "x"], ["a", "b", "c"])


@pytest.mark.parametrize("model,method,target", [
    ("rmu-e2b-bio-c81a100", "RMU", "wmdp-bio"), ("elm-g31-cyber-v3", "ELM", "wmdp-cyber"),
    ("r2mu-e2b-bio-control", "R2MU", "wmdp-bio"), ("rmu-gemma-cyber-lr25", "RMU", "wmdp-cyber"),
])
def test_a_checkpoint_names_its_method_and_target(model, method, target):
    assert (method_of(model), target_concept(model)) == (method, target)
    assert arm_of(Cell("r", model, target, "none-baseline", "on")) == "unlearned"
    assert arm_of(Cell("r", "gemma-4-E2B-it", target, "none-baseline", "on")) == "baseline"


def test_an_unlearning_run_is_taken_only_on_the_prompt_runs_instrument():
    prompt = {"gemma": {"h1"}, "llama": {"h9"}}
    assert same_instrument({"gemma": {"h1"}}, prompt)[0]
    ok, why = same_instrument({"llama": {"h4096"}}, prompt)       
    assert not ok and "h4096" in why
    assert not same_instrument({"olmo": {"x"}}, prompt)[0]


def test_the_default_draw_reads_every_unlearning_namespace_and_round_twos_names_parse():
    from scripts.plan_unlearn_round2 import commands
    from scripts.sample_panel import UNLEARN_RUNS
    from src.paths import UNLEARN_BASES
    assert set(UNLEARN_RUNS) == {"rmu1", *UNLEARN_BASES}
    for line in commands("train"):
        name = line.split("NAME=")[1].split(",")[0]
        assert method_of(name) and target_concept(name)


def _row(model, concept, item, prompt, *, think="on", cfg=None, probe="forget", answer="B"):
    trace = f"<|channel>thought\nweighing {concept} item {item}<channel|>" if think == "on" else ""
    return {"schema_version": "1", "git_commit": "x", "config_hash": cfg or f"h-{model}-{prompt}",
            "config_id": model, "model": model, "prompt_id": prompt,
            "condition": "baseline" if prompt == "none-baseline" else "suppressed",
            "think_state": think,
            "sampling": {"temperature": None, "top_p": None, "max_tokens": 16384, "seed": None},
            "item_id": f"{concept}-{item}", "concept_id": concept, "concept_class": "general",
            "concept_subtype": "hazard", "probe_type": probe,
            "raw_output": trace + f"Answer: {answer}\nWhy: a reason.", "raw_reasoning": None,
            "finish_reason": "stop"}


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def tree(tmp_path):


    g = tmp_path / "results"
    for model, think in (("gemma-a", "on"), ("llama-x", "na")):
        for prompt in ("none-baseline", "w1", "w2"):

            rows = [_row(model, c, i, prompt, think=think,
                         answer="C" if (prompt == "w1" and i == 0) else "B")
                    for c in CONCEPTS for i in range(6)]
            if think == "on":
                rows += [_row(model, c, i, prompt, think="off") for c in CONCEPTS for i in range(6)]
            rows += [_row(model, "wmdp-bio", 0, prompt, think=think, probe="neighbor")]
            _write(g / "p" / "generations" / f"{model}__{prompt}.jsonl", rows)
    for run, base_cfg in (("u1", "h-gemma-a-none-baseline"), ("u2", "another-config")):
        _write(g / run / "generations" / "gemma-a__none-baseline.jsonl",
               [_row("gemma-a", c, i, "none-baseline", cfg=base_cfg) for c in CONCEPTS
                for i in range(6)])
        for ckpt in (f"rmu-a-bio-{run}", f"elm-a-cyber-{run}"):
            _write(g / run / "generations" / f"{ckpt}__none-baseline.jsonl",
                   [_row(ckpt, c, i, "none-baseline") for c in CONCEPTS for i in range(6)])
    items = [{"item_id": f"{c}-{i}", "concept_id": c, "concept_class": "general",
              "concept_subtype": "hazard", "probe_type": "forget", "question": f"Q {c} {i}?",
              "answer_key": "B" if i < 4 else "A", "source": "WMDP",
              "meta": {"format": "mc", "choices": ["w", "x", "y", "z"]}}
             for c in CONCEPTS for i in range(6)]
    _write(tmp_path / "data" / "processed" / "p" / "items.jsonl", items)
    return tmp_path


def _sample(cwd, *argv):
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "sample_panel.py"),
                          "--prompt-run", "p", "--unlearn-runs", "u1", "u2", "--home", "h",
                          "--seed", "3", *argv], cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    return out


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_the_panel_covers_every_cell_and_only_on_target_checkpoints(tree):
    out = _sample(tree, "--per-concept", "2")
    d = tree / "results" / "h" / "judge_samples"
    assert sorted(p.name for p in d.glob("*.jsonl")) == [
        "panel-s3__p__gemma-a.jsonl", "panel-s3__p__llama-x.jsonl", "panel-s3__unlearned.jsonl"]
    gemma = _rows(d / "panel-s3__p__gemma-a.jsonl")
    assert len(gemma) == 3 * 2 * 2
    assert {r["think_state"] for r in gemma} == {"on"}     
    assert {r["probe_type"] for r in gemma} == {"forget"}
    assert {r["think_state"] for r in _rows(d / "panel-s3__p__llama-x.jsonl")} == {"na"}

    bio = {(r["prompt_id"], r["item_id"]) for r in gemma if r["concept_id"] == "wmdp-bio"}
    items = {i for _, i in bio}
    assert len(items) == 2 and len(bio) == 6
    unl = _rows(d / "panel-s3__unlearned.jsonl")
    assert {(r["model"], r["concept_id"]) for r in unl} == {("rmu-a-bio-u1", "wmdp-bio"),
                                                            ("elm-a-cyber-u1", "wmdp-cyber")}
    assert {r["item_id"] for r in unl if r["concept_id"] == "wmdp-bio"} == items
    assert "u2: LEFT OUT" in out.stdout
    m = json.loads((d / "panel-s3.manifest.json").read_text())
    arms = {(v["model"], v["arm"], v["method"], v["base_model"]) for v in m["rows"].values()}
    assert ("rmu-a-bio-u1", "unlearned", "RMU", "gemma-a") in arms
    assert ("gemma-a", "baseline", None, "gemma-a") in arms
    assert m["unlearn_runs"]["u2"]["included"] is False


def test_a_model_still_generating_can_be_left_out_and_added_later(tree):
    _sample(tree, "--per-concept", "2", "--skip-models", "llama-x")
    d = tree / "results" / "h" / "judge_samples"
    assert not (d / "panel-s3__p__llama-x.jsonl").exists()
    gemma = (d / "panel-s3__p__gemma-a.jsonl").read_text()
    _sample(tree, "--per-concept", "2", "--models", "llama-x")
    assert (d / "panel-s3__p__llama-x.jsonl").exists()
    assert (d / "panel-s3__p__gemma-a.jsonl").read_text() == gemma        
    m = json.loads((d / "panel-s3.manifest.json").read_text())
    assert {v["model"] for v in m["rows"].values()} >= {"gemma-a", "llama-x", "rmu-a-bio-u1"}


def _roster(cwd, roster: str):
    (cwd / "roster.yaml").write_text(roster)
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "sample_panel.py"),
                           "--prompt-run", "p", "--home", "h", "--roster", "roster.yaml"],
                          cwd=cwd, capture_output=True, text=True)


def test_a_roster_draws_exactly_what_it_names(tree):
    out = _roster(tree, "seed: 3\nper_concept: 2\nmodels: [gemma-a]\nprompts: [none-baseline, w1]\n"
                        "unlearn: {u1: [rmu-a-bio-u1], u2: [rmu-a-bio-u2]}\n")
    assert out.returncode == 0, out.stderr[-1500:]
    d = tree / "results" / "h" / "judge_samples"
    assert sorted(p.name for p in d.glob("*.jsonl")) == ["panel-s3__p__gemma-a.jsonl",
                                                         "panel-s3__unlearned.jsonl"]
    assert {r["prompt_id"] for r in _rows(d / "panel-s3__p__gemma-a.jsonl")} == {"none-baseline", "w1"}
    assert {r["model"] for r in _rows(d / "panel-s3__unlearned.jsonl")} == {"rmu-a-bio-u1"}
    assert "u2: LEFT OUT" in out.stdout
    m = json.loads((d / "panel-s3.manifest.json").read_text())
    assert m["history"][-1]["checkpoints"] == ["u1/rmu-a-bio-u1", "u2/rmu-a-bio-u2"]


@pytest.mark.parametrize("roster,why", [
    ("prompts: [none-baseline, w9]\n", "gemma-a × w9"),
    ("unlearn: {u1: [rmu-a-bio-u9]}\n", "u1/rmu-a-bio-u9"),
    ("unlearn: {u1: [elm-a-cyber-u1]}\nmodels: [gemma-a, qwen-z]\n", "qwen-z"),
    ("checkpionts: []\n", "unknown roster key"),
])
def test_a_roster_name_with_no_rows_is_refused_and_nothing_is_written(tree, roster, why):
    out = _roster(tree, "seed: 3\nper_concept: 2\n" + roster)
    assert out.returncode != 0 and why in out.stderr
    assert not (tree / "results" / "h").exists()


def test_the_committed_roster_names_real_prompts_runs_and_checkpoints():
    from scripts.sample_panel import UNLEARN_RUNS
    from src.prompts import ALL_PROMPT_IDS
    from src.prompts.labels import split_prompt_id
    roster = yaml.safe_load((ROOT / "configs/panels/panel1.yaml").read_text())
    assert (roster["name"], roster["per_concept"], roster["prompt_run"]) == ("panel1", 200, "wmdp8")
    assert all(split_prompt_id(p)[0] in ALL_PROMPT_IDS for p in roster["prompts"])
    assert "none-baseline" in roster["prompts"]
    assert set(roster["unlearn"]) <= set(UNLEARN_RUNS)
    for names in roster["unlearn"].values():
        assert all(method_of(n) and target_concept(n) for n in names)


def test_growing_the_panel_appends_and_a_changed_source_is_refused(tree):
    _sample(tree, "--per-concept", "2")
    d = tree / "results" / "h" / "judge_samples"
    f = d / "panel-s3__p__gemma-a.jsonl"
    before = f.read_text()
    out = _sample(tree, "--per-concept", "3")
    after = f.read_text()
    assert after.startswith(before) and len(_rows(f)) == 3 * 2 * 3
    m = json.loads((d / "panel-s3.manifest.json").read_text())
    assert m["files"][f.name] == {"rows": 18, "appended": 6, "per_concept": 3}
    assert [h["per_concept"] for h in m["history"]] == [2, 3]
    assert "GPU-hours to judge the" in out.stdout

    w1 = tree / "results" / "p" / "generations" / "gemma-a__w1.jsonl"
    first = _rows(f)[0]
    w1.write_text("".join(line + "\n" for line in w1.read_text().splitlines()
                          if json.loads(line)["item_id"] != first["item_id"]))
    refused = subprocess.run([sys.executable, str(ROOT / "scripts" / "sample_panel.py"),
                              "--prompt-run", "p", "--unlearn-runs", "u1", "--home", "h",
                              "--seed", "3", "--per-concept", "3"], cwd=tree,
                             capture_output=True, text=True)
    assert refused.returncode != 0 and "does not extend this file" in refused.stderr
    assert f.read_text() == after


def test_the_report_reads_a_checkpoint_as_its_own_arm(tree):
    _sample(tree, "--per-concept", "2")
    d = tree / "results" / "h"
    m = json.loads((d / "judge_samples" / "panel-s3.manifest.json").read_text())
    recs = []
    for gid, meta in m["rows"].items():
        for surface in ("trace", "explanation"):
            if surface == "trace" and meta["think_state"] != "on":
                continue
            recs.append({"generation_id": gid, "surface": surface, "concept_score": 2,
                         "framing_score": 0, "mention_score": 0, "judge_config_hash": "fp",
                         "framing_config_hash": "ff", "mention_config_hash": "fm",
                         "concept_raw": "Rating: [[2]]", "concept_finish_reason": "stop"})
    _write(d / "judgments" / "panel-s3__all.jsonl", recs)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "judge_sample_report.py"),
                          "--run", "h", "--sample", "panel-s3", "--spotcheck", "0"], cwd=tree,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    adir = d / "analysis" / "judge_sample" / "panel-s3"
    leak = list(csv.DictReader((adir / "leak.csv").open()))
    assert {r["condition"] for r in leak if r["model"] == "rmu-a-bio-u1"} == {"unlearned"}
    assert {r["condition"] for r in leak if r["model"] == "gemma-a"} == {"baseline", "suppressed"}
    control = list(csv.DictReader((adir / "control.csv").open()))
    n_base = sum(1 for v in m["rows"].values() if v["arm"] == "baseline")
    assert sum(int(r["n"]) for r in control) == n_base + sum(
        1 for v in m["rows"].values() if v["arm"] == "baseline" and v["think_state"] == "on")
    assert "panel" in (adir / "report.md").read_text()
    cells = list(csv.DictReader((adir / "cells.csv").open()))      
    keys = {(r["model"], r["prompt_id"], r["concept_id"], r["surface"]) for r in cells}
    assert ("gemma-a", "w1", "wmdp-bio", "trace") in keys
    assert ("rmu-a-bio-u1", "none-baseline", "wmdp-bio", "explanation") in keys
    assert len(keys) == len(cells)


def test_the_join_puts_everything_about_a_generation_in_one_record(tree):


    _sample(tree, "--per-concept", "6")
    d = tree / "results" / "h"
    m = json.loads((d / "judge_samples" / "panel-s3.manifest.json").read_text())
    recs = []
    for gid, meta in m["rows"].items():
        for surface in ("trace", "explanation"):
            if surface == "trace" and meta["think_state"] != "on":
                continue
            odd = meta["item_id"] == "wmdp-bio-5" and meta["prompt_id"] == "w2"
            recs.append({"generation_id": gid, "surface": surface,
                         "concept_score": 0 if odd else 2, "framing_score": 0,
                         "mention_score": 2 if odd else 0, "judge_config_hash": "fp",
                         "framing_config_hash": "ff", "mention_config_hash": "fm",
                         "concept_raw": "<think>hm</think>The knowledge is there.\nRating: [[2]]",
                         "concept_finish_reason": "stop"})
    _write(d / "judgments" / "panel-s3__all.jsonl", recs)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "join_judged.py"), "--run", "h",
                          "--sample", "panel-s3"], cwd=tree, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    rows = [json.loads(line) for line in
            (d / "analysis" / "judge_sample" / "panel-s3" / "joined.jsonl").read_text().splitlines()]
    assert len(rows) == len(m["rows"])
    flipped = next(r for r in rows if r["model"] == "gemma-a" and r["prompt_id"] == "w1"
                   and r["item_id"] == "wmdp-bio-0")
    assert (flipped["question"], flipped["answer_key"], flipped["answer_text"]) == \
        ("Q wmdp-bio 0?", "B", "x")
    assert (flipped["model_answer"], flipped["correct"]) == ("C", False)
    assert (flipped["baseline_answer"], flipped["baseline_correct"]) == ("B", True)
    assert flipped["outcome"] == "removed" and flipped["baseline_generation_id"]
    assert "weighing wmdp-bio item 0" in flipped["trace"] and flipped["explanation"] == "a reason."
    assert flipped["trace_concept"] == 2 and flipped["trace_concept_why"] == \
        "The knowledge is there.\nRating: [[2]]"                      
    assert flipped["trace_cell"] == "used_unnamed"
    held = {r["outcome"] for r in rows if r["arm"] == "suppressed"}
    assert held == {"removed", "held", "no_headroom"}
    assert {r["outcome"] for r in rows if r["arm"] == "baseline"} == {None}
    ckpt = [r for r in rows if r["model"] == "rmu-a-bio-u1"]          
    assert ckpt and all(r["baseline_correct"] is not None for r in ckpt)
    assert {r["outcome"] for r in ckpt} == {"held", "no_headroom"}
    odd = [r for r in rows if r["explanation_stage1_0_but"]]
    assert odd and all(r["explanation_stage1_0_but"] == "used,named" for r in odd)
    assert "stage 1 = 0 but a later stage disagrees" in out.stdout
    header = (d / "analysis" / "judge_sample" / "panel-s3" / "joined.csv").read_text().splitlines()[0]
    assert "question" in header and "trace_mention_why" in header and ",trace," not in header
