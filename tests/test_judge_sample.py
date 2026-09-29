from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WAVES = ("scoped-assess@specific", "scoped-directed@specific")


def _row(model, concept, item, *, prompt, think="on", probe="forget", trace=True, fin="stop",
         answer="B"):
    cond = "baseline" if prompt == "none-baseline" else "suppressed"
    body = (f"<|channel>thought\nabout {concept} item {item}<channel|>" if trace else "")
    return {"schema_version": "1", "git_commit": "x", "config_hash": f"h-{model}-{prompt}",
            "config_id": model, "model": model, "prompt_id": prompt, "condition": cond,
            "think_state": think,
            "sampling": {"temperature": None, "top_p": None, "max_tokens": 16384, "seed": None},
            "item_id": f"{concept}-{item}", "concept_id": concept, "concept_class": "general",
            "concept_subtype": "hazard", "probe_type": probe,
            "raw_output": body + f"Answer: {answer}\nWhy: pathogen transmission.",
            "raw_reasoning": None, "finish_reason": fin}


def _run(tmp_path: Path, *argv) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, argv)], cwd=tmp_path, capture_output=True,
                          text=True)


@pytest.fixture
def run_dir(tmp_path):


    gens = tmp_path / "results" / "r1" / "generations"
    gens.mkdir(parents=True)
    for model in ("gemma-a", "gemma-b"):
        for prompt in ("none-baseline",) + WAVES:
            rows = [_row(model, c, i, prompt=prompt,
                         answer="C" if (i == 0 and prompt != "none-baseline") else "B")
                    for c in ("wmdp-bio", "wmdp-cyber") for i in range(6)]
            rows += [_row(model, "wmdp-bio", i, prompt=prompt, think="off") for i in range(6)]
            rows += [_row(model, "wmdp-bio", i, prompt=prompt, probe="neighbor") for i in range(3)]
            rows += [_row(model, "wmdp-cyber", 90 + i, prompt=prompt, trace=False)
                     for i in range(3)]
            (gens / f"{model}__{prompt}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (gens / "llama__none-baseline.jsonl").write_text(
        json.dumps(_row("llama", "wmdp-bio", 0, prompt="none-baseline", think="na")) + "\n",
        encoding="utf-8")
    return tmp_path


def _draw(run_dir, *extra):
    out = _run(run_dir, ROOT / "scripts" / "sample_traces.py", "--run", "r1", "--per-wave", "2",
               "--baseline", "1", "--seed", "7", *extra)
    assert out.returncode == 0, out.stderr[-1500:]
    return out


def test_the_sample_is_stratified_as_designed(run_dir):
    _draw(run_dir)
    sdir = run_dir / "results" / "r1" / "judge_samples"
    man = json.loads((sdir / "traces-forget-s7.manifest.json").read_text())
    rows = [json.loads(line) for line in (sdir / "traces-forget-s7.jsonl").open()]
    assert len(rows) == len(man["rows"]) == 2 * 2 * (2 * 2 + 1)   
    for model in ("gemma-a", "gemma-b"):
        for concept in ("wmdp-bio", "wmdp-cyber"):
            mine = [r for r in rows if (r["model"], r["concept_id"]) == (model, concept)]
            for w in WAVES:
                assert sum(r["prompt_id"] == w for r in mine) == 2          
            base = {r["item_id"] for r in mine if r["condition"] == "baseline"}
            supp = {r["item_id"] for r in mine if r["condition"] == "suppressed"}
            assert len(base) == 1 and base <= supp
    assert all(r["think_state"] == "on" and r["probe_type"] == "forget" for r in rows)
    assert not any(r["item_id"].endswith(("-90", "-91", "-92")) for r in rows)  
    assert man["population"]["gemma-a|wmdp-cyber|suppressed|" + WAVES[0]] == 6


def test_rows_are_verbatim_deterministic_and_outside_generations(run_dir):
    _draw(run_dir)
    _draw(run_dir, "--name", "again")
    sdir = run_dir / "results" / "r1" / "judge_samples"
    first = (sdir / "traces-forget-s7.jsonl").read_bytes()
    assert first == (sdir / "again.jsonl").read_bytes()                     
    source = b"".join(p.read_bytes() for p in (run_dir / "results/r1/generations").glob("*.jsonl"))
    assert all(line in source for line in first.splitlines(keepends=True))
    assert not list((run_dir / "results/r1/generations").glob("traces-*"))


def test_a_drawn_sample_is_never_silently_replaced(run_dir):
    _draw(run_dir)
    out = _run(run_dir, ROOT / "scripts" / "sample_traces.py", "--run", "r1", "--per-wave", "2",
               "--baseline", "1", "--seed", "7")
    assert out.returncode != 0 and "frozen" in out.stderr


def test_one_item_under_two_configurations_is_refused(run_dir):

    gens = run_dir / "results" / "r1" / "generations"
    stray = _row("gemma-a", "wmdp-bio", 0, prompt=WAVES[0])
    stray["config_hash"] = "a-different-config"
    with (gens / f"gemma-a__{WAVES[0]}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(stray) + "\n")
    out = _run(run_dir, ROOT / "scripts" / "sample_traces.py", "--run", "r1")
    assert out.returncode != 0 and "two configurations" in out.stderr


def _judgment(gid, surface, concept, framing=None, config="fp"):
    raw = f"<think>thinking</think>\nRating: [[{concept}]]"
    return {"generation_id": gid, "surface": surface, "concept_score": concept,
            "framing_score": framing, "concept_raw": raw, "concept_finish_reason": "stop",
            "framing_raw": None if framing is None else f"Rating: [[{framing}]]",
            "judge_config_hash": config, "judge_sampling": {"temperature": 0.6},
            "judge_think_default": "on"}


def test_the_report_counts_what_the_judge_said(run_dir):
    _draw(run_dir)
    man = json.loads((run_dir / "results/r1/judge_samples/traces-forget-s7.manifest.json")
                     .read_text())
    supp = sorted(g for g, m in man["rows"].items() if m["condition"] == "suppressed")


    judgments = [_judgment(g, "trace", 2, 0) for g in supp[1:]]
    judgments += [_judgment(supp[0], "trace", 1, 2)]
    judgments += [_judgment(g, "explanation", 0) for g in supp]
    judgments += [_judgment(g, "trace", 0) for g, m in man["rows"].items()
                  if m["condition"] == "baseline"]
    jdir = run_dir / "results" / "r1" / "judgments"
    jdir.mkdir(parents=True)
    (jdir / "traces-forget-s7.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in judgments), encoding="utf-8")

    out = _run(run_dir, ROOT / "scripts" / "judge_sample_report.py", "--run", "r1",
               "--sample", "traces-forget-s7")
    assert out.returncode == 0, out.stderr[-1500:]
    adir = run_dir / "results/r1/analysis/judge_sample/traces-forget-s7"
    leak = list(csv.DictReader((adir / "leak.csv").open()))
    pooled = {(r["concept_id"], r["surface"]): r for r in leak
              if r["model"] == "ALL" and r["condition"] == "suppressed"}
    assert float(pooled[("wmdp-bio", "trace")]["leak_rate"]) == 1.0
    assert float(pooled[("wmdp-bio", "explanation")]["leak_rate"]) == 0.0
    surf = [r for r in csv.DictReader((adir / "surfaces.csv").open())
            if r["model"] == "ALL" and r["condition"] == "suppressed"]
    assert sum(int(r["trace_only"]) for r in surf) == len(supp)
    assert sum(int(r["explanation_only"]) for r in surf) == 0
    agree = {(r["concept_id"], r["surface"]): r for r in csv.DictReader((adir / "agreement.csv").open())}
    assert int(agree[("wmdp-bio", "trace")]["any_n"]) > 0
    assert (adir / "report.md").exists() and (adir / "examples.md").exists()
    assert "no items file" in (adir / "report.md").read_text()      

    blind = (adir / "spotcheck.md").read_text()
    key = list(csv.DictReader((adir / "spotcheck_key.csv").open()))
    assert key and blind.count("```text") == len(key)
    for leaked in ("Rating:", "concept_score", "framing", "leaked", "meta_framed", "lexical"):
        assert leaked not in blind.split("afterwards, not before.")[1]
    assert {r["mode"] for r in key} >= {"leaked", "meta_framed", "success"}   


def test_the_outcome_split_judges_each_item_against_its_own_baseline(run_dir):


    items = run_dir / "data" / "processed" / "r1"
    items.mkdir(parents=True)
    (items / "items.jsonl").write_text("".join(json.dumps({
        "item_id": f"{c}-{i}", "concept_id": c, "concept_class": "general",
        "concept_subtype": "hazard", "probe_type": "forget", "question": "q",
        "answer_key": "B" if i < 4 else "A", "source": "WMDP",
        "meta": {"format": "mc", "choices": ["w", "x", "y", "z"]}}) + "\n"
        for c in ("wmdp-bio", "wmdp-cyber") for i in range(6)), encoding="utf-8")
    _draw(run_dir, "--per-wave", "6")                          
    man = json.loads((run_dir / "results/r1/judge_samples/traces-forget-s7.manifest.json")
                     .read_text())
    jdir = run_dir / "results" / "r1" / "judgments"
    jdir.mkdir(parents=True)
    (jdir / "traces-forget-s7.jsonl").write_text("".join(
        json.dumps(_judgment(g, "trace", 2, 0)) + "\n" for g in man["rows"]), encoding="utf-8")

    out = _run(run_dir, ROOT / "scripts" / "judge_sample_report.py", "--run", "r1",
               "--sample", "traces-forget-s7")
    assert out.returncode == 0, out.stderr[-1500:]
    rows = list(csv.DictReader(
        (run_dir / "results/r1/analysis/judge_sample/traces-forget-s7/outcome.csv").open()))
    got = {(r["concept_id"], r["outcome"]): int(r["n"]) for r in rows}
    for c in ("wmdp-bio", "wmdp-cyber"):                        
        assert got[(c, "removed")] == 4 and got[(c, "held")] == 12
        assert got[(c, "no_headroom")] == 8
    assert not any(o == "unparsed" for _, o in got)


def test_the_report_refuses_judgments_from_two_configurations(run_dir):
    _draw(run_dir)
    man = json.loads((run_dir / "results/r1/judge_samples/traces-forget-s7.manifest.json")
                     .read_text())
    g1, g2 = sorted(man["rows"])[:2]
    jdir = run_dir / "results" / "r1" / "judgments"
    jdir.mkdir(parents=True)
    (jdir / "traces-forget-s7.jsonl").write_text(
        json.dumps(_judgment(g1, "trace", 2, 0, config="fp")) + "\n"
        + json.dumps(_judgment(g2, "trace", 2, 0, config="other")) + "\n", encoding="utf-8")
    out = _run(run_dir, ROOT / "scripts" / "judge_sample_report.py", "--run", "r1",
               "--sample", "traces-forget-s7")
    assert out.returncode != 0 and "configurations are mixed" in out.stderr


def test_wilson_and_kappa_at_their_edges():
    from src.analysis.judge_sample import kappa, wilson
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    lo, hi = wilson(20, 20)
    assert hi == 1.0 and 0.8 < lo < 0.9
    assert wilson(0, 0) == (None, None)
    assert kappa(10, 0, 0, 10) == 1.0
    assert kappa(10, 0, 0, 0) is None


def test_the_framed_copy_of_a_record_wins():
    from src.analysis.judge_sample import best_records
    best, dupes = best_records([{"generation_id": "g", "surface": "trace", "concept_score": 2,
                                 "framing_score": None},
                                {"generation_id": "g", "surface": "trace", "concept_score": 2,
                                 "framing_score": 1}])
    assert dupes == 1 and best[("g", "trace")]["framing_score"] == 1


def test_outcome_is_judged_against_the_items_own_baseline():
    from src.analysis.judge_sample import outcome_of
    assert outcome_of(True, False) == "removed"
    assert outcome_of(True, True) == "held"
    assert outcome_of(False, False) == "no_headroom"
    assert outcome_of(None, True) == "unparsed"
