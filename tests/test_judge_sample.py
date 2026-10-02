from __future__ import annotations

import csv
from collections import Counter
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


def _judgment(gid, surface, concept, framing=None, mention=None, config="fp"):
    raw = f"<think>thinking</think>\nRating: [[{concept}]]"
    rec = {"generation_id": gid, "surface": surface, "concept_score": concept,
           "framing_score": framing, "concept_raw": raw, "concept_finish_reason": "stop",
           "framing_raw": None if framing is None else f"Rating: [[{framing}]]",
           "judge_config_hash": config, "judge_sampling": {"temperature": 0.6},
           "judge_think_default": "on"}
    if framing is not None:
        rec["framing_config_hash"] = "ff"
    if mention is not None:
        rec.update(mention_score=mention, mention_raw=f"Rating: [[{mention}]]",
                   mention_config_hash="fm")
    return rec


def _write_judgments(run_dir, judgments, name="traces-forget-s7.jsonl"):
    jdir = run_dir / "results" / "r1" / "judgments"
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / name).write_text("".join(json.dumps(j) + "\n" for j in judgments), encoding="utf-8")


def _report(run_dir, *extra):
    return _run(run_dir, ROOT / "scripts" / "judge_sample_report.py", "--run", "r1",
                "--sample", "traces-forget-s7", *extra)


def _manifest(run_dir):
    return json.loads((run_dir / "results/r1/judge_samples/traces-forget-s7.manifest.json")
                      .read_text())


ADIR = "results/r1/analysis/judge_sample/traces-forget-s7"


def test_the_report_counts_what_the_judge_said(run_dir):
    _draw(run_dir)
    man = _manifest(run_dir)
    supp = sorted(g for g, m in man["rows"].items() if m["condition"] == "suppressed")


    judgments = [_judgment(g, "trace", 2, 0, 2) for g in supp[1:]]
    judgments += [_judgment(supp[0], "trace", 1, 2, 2)]
    judgments += [_judgment(g, "explanation", 0, mention=0) for g in supp]
    judgments += [_judgment(g, "trace", 0, mention=0) for g, m in man["rows"].items()
                  if m["condition"] == "baseline"]
    _write_judgments(run_dir, judgments)

    out = _report(run_dir)
    assert out.returncode == 0, out.stderr[-1500:]
    adir = run_dir / ADIR
    leak = list(csv.DictReader((adir / "leak.csv").open()))
    pooled = {(r["concept_id"], r["surface"]): r for r in leak
              if r["model"] == "ALL" and r["condition"] == "suppressed"}
    assert float(pooled[("wmdp-bio", "trace")]["leak_rate"]) == 1.0
    assert float(pooled[("wmdp-bio", "explanation")]["leak_rate"]) == 0.0

    n_bio = int(pooled[("wmdp-bio", "trace")]["n"])
    bio_meta = int(supp[0] in {g for g, m in man["rows"].items() if m["concept_id"] == "wmdp-bio"})
    assert int(pooled[("wmdp-bio", "trace")]["meta_framed"]) == bio_meta
    assert int(pooled[("wmdp-bio", "trace")]["leaked"]) == n_bio - bio_meta
    assert float(pooled[("wmdp-bio", "explanation")]["success_frac"]) == 1.0
    assert float(pooled[("wmdp-bio", "trace")]["mention_coverage"]) == 1.0
    surf = [r for r in csv.DictReader((adir / "surfaces.csv").open())
            if r["model"] == "ALL" and r["condition"] == "suppressed"]
    assert sum(int(r["trace_only"]) for r in surf) == len(supp)
    assert sum(int(r["explanation_only"]) for r in surf) == 0
    agree = {(r["concept_id"], r["surface"]): r for r in csv.DictReader((adir / "agreement.csv").open())}
    assert int(agree[("wmdp-bio", "trace")]["any_n"]) > 0
    assert (adir / "report.md").exists() and (adir / "examples.md").exists()
    assert "no items file" in (adir / "report.md").read_text()      
    assert "## The standing control" in (adir / "report.md").read_text()

    blind = (adir / "spotcheck-trace-s7-n50.md").read_text()
    key = list(csv.DictReader((adir / "spotcheck-trace-s7-n50.key-fp-ff-fm.csv").open()))
    assert key and blind.count("```text") == len(key) == len(man["rows"])   
    passages = blind.split("\n---\n", 1)[1]
    for leaked in ("Rating:", "concept_score", "framing", "leaked", "meta_framed", "lexical",
                   "mention_score"):
        assert leaked not in passages
    assert "biology (including biosecurity)" in passages and "names: biosecurity: " in passages
    assert (adir / "spotcheck-explanation-s7-n50.md").exists()


def test_the_spot_check_draw_ignores_the_judge_and_a_human_s_labels_survive(run_dir):


    _draw(run_dir)
    man = _manifest(run_dir)
    _write_judgments(run_dir, [_judgment(g, "trace", 2, 0, 2) for g in man["rows"]])
    assert _report(run_dir, "--spotcheck", "6").returncode == 0
    adir = run_dir / ADIR
    blind = adir / "spotcheck-trace-s7-n6.md"
    first = blind.read_text()
    blind.write_text(first + "\nMY LABELS: 1 used, 2 used\n")           


    _write_judgments(run_dir, [dict(_judgment(g, "trace", 1, 2, 0), framing_config_hash="ff2")
                               for g in man["rows"]])
    assert _report(run_dir, "--spotcheck", "6").returncode == 0
    assert blind.read_text().endswith("MY LABELS: 1 used, 2 used\n")      
    old = list(csv.DictReader((adir / "spotcheck-trace-s7-n6.key-fp-ff-fm.csv").open()))
    new = list(csv.DictReader((adir / "spotcheck-trace-s7-n6.key-fp-ff2-fm.csv").open()))
    assert [r["generation_id"] for r in old] == [r["generation_id"] for r in new]   
    assert {r["framing_score"] for r in old} == {"0"} and {r["framing_score"] for r in new} == {"2"}

    from src.analysis.spotcheck import spotcheck_ids
    ids = spotcheck_ids(man["rows"], "trace", 6, 7, "traces-forget-s7")
    strata = [(man["rows"][g]["concept_id"], man["rows"][g]["condition"]) for g in ids]
    assert len(set(strata)) == 4 and max(strata.count(s) for s in set(strata)) == 2

    assert _report(run_dir, "--spotcheck", "4").returncode == 0
    assert (adir / "spotcheck-trace-s7-n4.md").exists()
    assert blind.read_text().endswith("MY LABELS: 1 used, 2 used\n")


def test_the_label_audit_draws_every_judged_label(run_dir):

    _draw(run_dir)
    man = _manifest(run_dir)
    gids = sorted(man["rows"])
    labels = [(2, 0, 2), (2, 0, 0), (0, None, 2), (0, None, 0), (2, 2, 2)]
    _write_judgments(run_dir, [_judgment(g, "trace", *labels[i % len(labels)])
                               for i, g in enumerate(gids)])
    assert _report(run_dir, "--audit", "2", "--spotcheck", "0").returncode == 0
    adir = run_dir / ADIR
    key = list(csv.DictReader((adir / "audit-trace-fp-ff-fm.key.csv").open()))
    got = Counter(r["label"] for r in key)
    assert got == {"leaked_direct": 2, "leaked_inadvertent": 2, "meta_framed": 2, "success": 2,
                   "contradiction": 2}
    assert not list(adir.glob("spotcheck-*"))                       


def test_the_baseline_is_a_standing_control(run_dir):


    _draw(run_dir, "--baseline", "3")
    man = _manifest(run_dir)
    base = sorted(g for g, m in man["rows"].items() if m["condition"] == "baseline"
                  and m["concept_id"] == "wmdp-bio")
    assert len(base) >= 3
    judgments = [_judgment(base[0], "trace", 1, 2, 2), _judgment(base[1], "trace", 2, 2, 2)]
    judgments += [_judgment(g, "trace", 2, 0, 2) for g in base[2:]]
    _write_judgments(run_dir, judgments)
    out = _report(run_dir)
    assert out.returncode == 0, out.stderr[-1500:]
    control = {(r["concept_id"], r["surface"]): r
               for r in csv.DictReader((run_dir / ADIR / "control.csv").open())}
    bio = control[("wmdp-bio", "trace")]
    framed = len(base) - 1
    assert float(bio["named_only_share"]) == round(1 / framed, 4)
    assert int(bio["n_contradiction"]) == 1
    assert "standing control" in out.stdout and "contradictions 1" in out.stdout


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


def test_the_report_refuses_a_half_finished_mention_run(run_dir):
    _draw(run_dir)
    g1, g2 = sorted(_manifest(run_dir)["rows"])[:2]
    _write_judgments(run_dir, [dict(_judgment(g1, "trace", 2, 0, 2), mention_config_hash="old"),
                               _judgment(g2, "trace", 2, 0, 2)])
    out = _report(run_dir)
    assert out.returncode != 0 and "mention rubrics are mixed" in out.stderr


def test_the_report_refuses_a_half_finished_re_frame(run_dir):


    _draw(run_dir)
    man = json.loads((run_dir / "results/r1/judge_samples/traces-forget-s7.manifest.json")
                     .read_text())
    g1, g2 = sorted(man["rows"])[:2]
    old = dict(_judgment(g1, "trace", 2, 2), framing_config_hash="old-rubric")
    new = dict(_judgment(g2, "trace", 2, 0), framing_config_hash="new-rubric")
    jdir = run_dir / "results" / "r1" / "judgments"
    jdir.mkdir(parents=True)
    (jdir / "traces-forget-s7.jsonl").write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n",
                                                 encoding="utf-8")
    out = _run(run_dir, ROOT / "scripts" / "judge_sample_report.py", "--run", "r1",
               "--sample", "traces-forget-s7")
    assert out.returncode != 0 and "framing rubrics are mixed" in out.stderr


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


def test_a_trace_is_incoherent_only_if_it_was_cut_at_the_cap_while_looping():


    from scripts.check_truncation import LOOP_AT, loop_score
    from src.analysis.judge_sample import is_incoherent, joined
    loop = "<|channel>thought\n" + "so the answer must be the one that I keep coming back to " * 80
    fresh = "<|channel>thought\n" + " ".join(f"step {i} considers option {i % 4}" for i in range(400))
    assert is_incoherent({"finish_reason": "length", "raw_output": loop}, loop_score, LOOP_AT)
    assert not is_incoherent({"finish_reason": "stop", "raw_output": loop}, loop_score, LOOP_AT)
    assert not is_incoherent({"finish_reason": "length", "raw_output": fresh}, loop_score, LOOP_AT)
    best = {("g", s): {"generation_id": "g", "surface": s, "concept_score": 0, "mention_score": 2}
            for s in ("trace", "explanation")}
    rows, _ = joined(best, {"g": {"concept_id": "wmdp-bio"}}, {"g"})
    modes = {r["surface"]: r["mode"] for r in rows}
    assert modes == {"trace": "incoherent", "explanation": "meta_framed"}


def test_framing_drift_cross_tabs_only_rows_scored_under_both_rubrics():
    from scripts.framing_drift import drift

    def rec(gid, cond, prompt, score, rubric):
        return {"generation_id": gid, "surface": "trace", "condition": cond, "prompt_id": prompt,
                "framing_score": score, "framing_config_hash": rubric}
    table = drift([rec("a", "suppressed", "scoped-assess-decoy-morph@specific", 2, "v2"),
                   rec("a", "suppressed", "scoped-assess-decoy-morph@specific", 0, "v3"),
                   rec("b", "baseline", "none-baseline", 0, "v2"),
                   rec("b", "baseline", "none-baseline", 0, "v3"),
                   rec("c", "suppressed", "scoped-assess@specific", 1, "v2")],   
                  "framing", "v2", "v3")
    assert table == {("trace", "decoy"): Counter({(2, 0): 1}),
                     ("trace", "baseline"): Counter({(0, 0): 1})}


def test_outcome_is_judged_against_the_items_own_baseline():
    from src.analysis.judge_sample import outcome_of
    assert outcome_of(True, False) == "removed"
    assert outcome_of(True, True) == "held"
    assert outcome_of(False, False) == "no_headroom"
    assert outcome_of(None, True) == "unparsed"
