from __future__ import annotations

from pathlib import Path

import yaml

from src.paths import (analysis_dir, generations_dir, generations_glob, items_path,
                       judgments_dir, primitive_judgments_dir)

_DIRS = (items_path, generations_dir, judgments_dir, primitive_judgments_dir, analysis_dir)


def test_every_artifact_path_is_namespaced_by_run():
    for fn in _DIRS:
        assert "exp1" in str(fn("exp1"))
        assert "round1" in str(fn("round1"))


def test_the_two_studies_never_collide():
    for fn in _DIRS:
        assert fn("exp1") != fn("round1"), f"{fn.__name__} collides across runs"


def test_items_path_shape():
    assert items_path("exp1") == Path("data/processed/exp1/items.jsonl")


def test_generations_glob_matches_its_dir():
    assert generations_glob("exp1").startswith(str(generations_dir("exp1")))


def test_configs_declare_their_run():

    for cfg_name, want in (("exp1.yaml", "exp1"), ("screening.yaml", "round1")):
        cfg = yaml.safe_load((Path("configs") / cfg_name).read_text())
        assert cfg.get("run") == want, f"configs/{cfg_name} must declare `run: {want}`"


def test_a_dry_run_never_overwrites_a_real_optimiser_result():


    from src.opt import result_path

    real = result_path("wmdp1", "Qwen3-8B", "A", dry_run=False)
    smoke = result_path("wmdp1", "Qwen3-8B", "A", dry_run=True)
    assert real != smoke
    assert real.name == "Qwen3-8B__A.json" and smoke.name == "Qwen3-8B__A.dryrun.json"
    assert real.parent == smoke.parent == Path("results/wmdp1/optimize")


def test_the_dspy_arm_writes_outside_every_manual_run_tree():


    from src.opt import result_path
    from src.paths import RESULTS_ROOT

    for src_run in ("exp1", "wmdp1"):
        out = result_path(f"dspy-{src_run}", "M", "A", dry_run=False)
        assert not str(out).startswith(str(RESULTS_ROOT / src_run) + "/"), \
            f"{out} lands inside the manual run {src_run}"
        assert out.parent == RESULTS_ROOT / f"dspy-{src_run}" / "optimize"


def test_stale_analysis_flags_a_scoring_older_than_its_data(tmp_path, monkeypatch):


    import os
    import time

    import src.paths as P

    monkeypatch.setattr(P, "RESULTS_ROOT", tmp_path)
    gens = tmp_path / "r" / "generations"
    ana = tmp_path / "r" / "analysis"
    gens.mkdir(parents=True)
    ana.mkdir(parents=True)

    old = ana / "accuracy.csv"
    old.write_text("model\n")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    (gens / "M__none-baseline.jsonl").write_text("{}\n")          

    assert P.stale_analysis("r") == [old]

    fresh = ana / "surface_gap.csv"                                
    fresh.write_text("model\n")
    assert fresh not in P.stale_analysis("r")


def test_stale_analysis_is_empty_when_there_are_no_generations(tmp_path, monkeypatch):
    import src.paths as P

    monkeypatch.setattr(P, "RESULTS_ROOT", tmp_path)
    (tmp_path / "r" / "analysis").mkdir(parents=True)
    (tmp_path / "r" / "analysis" / "accuracy.csv").write_text("model\n")
    assert P.stale_analysis("r") == []


def test_optimize_namespace_is_round_then_mode():


    from src.paths import optimize_namespace
    assert optimize_namespace(1, prompt_model="Qwen3-14B") == "dspy-r1"
    assert optimize_namespace(1) == "dspy-r1-self"
    assert optimize_namespace(2, demos=4) == "dspy-r2-self-fs4"
    assert optimize_namespace(2, prompt_model="Qwen3-14B", demos=4) == "dspy-r2-fs4"


def test_the_two_modes_never_share_a_namespace_within_a_round():


    from src.paths import optimize_namespace
    names = {optimize_namespace(1, prompt_model=pm, demos=d)
             for pm in (None, "Qwen3-14B") for d in (0, 4)}
    assert len(names) == 4


def test_a_round_is_never_the_plan_run():


    from src.paths import RUNS, optimize_namespace
    assert optimize_namespace(1) != "round1"
    assert "round1" in RUNS


def test_round_zero_and_negative_are_refused():
    import pytest as _pytest
    from src.paths import optimize_namespace
    for bad in (0, -1):
        with _pytest.raises(ValueError, match="round must be"):
            optimize_namespace(bad)
