from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.optimize_table import discordance, mcnemar_exact, summarise, verdict  # noqa: E402


def test_only_discordant_pairs_count():


    seed = {"a": 1, "b": 1, "c": 0, "d": 0}
    found = {"a": 1, "b": 0, "c": 1, "d": 0}
    assert discordance(seed, found) == (1, 1, 4)


def test_a_tie_is_reported_as_no_difference_not_as_a_win():
    seed_only, found_only, _ = 1, 1, 2
    p = mcnemar_exact(seed_only, found_only)
    assert p == 1.0
    assert "no reliable difference" in verdict(seed_only, found_only, p)


def test_a_three_item_gain_on_a_hundred_items_is_not_significant():


    p = mcnemar_exact(seed_only=0, found_only=3)
    assert p is not None and p > 0.05
    assert "no reliable difference" in verdict(0, 3, p)


def test_a_large_one_sided_gain_is_significant():
    p = mcnemar_exact(seed_only=1, found_only=12)
    assert p < 0.05 and "found BEATS seed" in verdict(1, 12, p)


def test_a_loss_is_named_as_a_loss():
    p = mcnemar_exact(seed_only=12, found_only=1)
    assert "found is WORSE" in verdict(12, 1, p)


def test_identical_runs_say_the_search_changed_nothing():
    items = {"a": 1, "b": 0}
    assert mcnemar_exact(*discordance(items, items)[:2]) is None
    assert "changed nothing" in verdict(0, 0, None)


def _write(tmp_path, **over):
    run = {"model": "M", "objective": "A", "concepts": ["wmdp-bio"], "think_state": "off",
           "seed_prompt": "scoped-world@generic", "n_dev": 100, "dry_run": False,
           "seed_score": 5.0, "found_score": 8.0,
           "seed_clause": "SEED CLAUSE.", "found_clause": "FOUND CLAUSE.",
           "seed_items": {f"i{n}": int(n < 5) for n in range(100)},
           "found_items": {f"i{n}": int(n < 8) for n in range(100)}}
    run.update(over)
    d = tmp_path / "optimize"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"M__{run['objective']}{'.dryrun' if run['dry_run'] else ''}.json").write_text(
        json.dumps(run), encoding="utf-8")
    return d


def test_summarise_pairs_the_items_rather_than_differencing_the_means(tmp_path):
    d = _write(tmp_path)
    row = summarise(json.loads(next(d.glob("*.json")).read_text()))
    assert row["delta"] == 3.0 and row["gained"] == 3 and row["lost"] == 0
    assert row["p_value"] > 0.05, "3 of 100 is not a result"


def test_the_page_renders_and_shows_both_clauses(tmp_path):
    d = _write(tmp_path)
    out = tmp_path / "optimize.md"
    r = subprocess.run([sys.executable, "scripts/present_optimize.py", "--optimize-dir", str(d),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "SEED CLAUSE." in text and "FOUND CLAUSE." in text
    assert "one item is one point" in text
    assert "not a result" in text, "the val-candidate caveat must survive"


def test_a_dry_run_is_quarantined_from_the_scoreboard(tmp_path):
    d = _write(tmp_path, dry_run=True, found_score=5.0,
               found_items={f"i{n}": int(n < 5) for n in range(100)},
               found_clause="SEED CLAUSE.")
    out = tmp_path / "optimize.md"
    r = subprocess.run([sys.executable, "scripts/present_optimize.py", "--optimize-dir", str(d),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "No completed optimiser runs yet" in text
    assert "wiring checks, not comparisons" in text


def test_a_returned_seed_is_only_a_finding_if_the_search_actually_explored():


    from src.analysis.optimize_table import search_note

    real = {"n_candidates": 10, "n_distinct_clauses": 9, "n_trials": 6,
            "trial_scores": [3.0, 5.0, 5.0], "total_lm_calls": 4200}
    note = search_note(real, found_is_seed=True)
    assert "9 distinct clauses" in note and "none of them beat it" in note

    noop = {"n_candidates": 1, "n_distinct_clauses": 1, "n_trials": 1, "trial_scores": [5.0]}
    assert "barely explored" in search_note(noop, found_is_seed=True)

    missing = search_note({}, found_is_seed=True)
    assert "cannot be read as" in missing, "an un-instrumented run must say so"
    assert search_note({}, found_is_seed=False) == ""


def test_the_markdown_page_survives_a_missing_matplotlib(tmp_path, monkeypatch):


    d = _write(tmp_path)
    out = tmp_path / "optimize.md"
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)}

    blocker = tmp_path / "sitecustomize.py"
    blocker.write_text("import sys\nsys.modules['matplotlib'] = None\n", encoding="utf-8")
    env["PYTHONPATH"] = f"{tmp_path}:{ROOT}"
    r = subprocess.run([sys.executable, "scripts/present_optimize.py", "--optimize-dir", str(d),
                        "--out", str(out)], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert out.exists() and "FOUND CLAUSE." in out.read_text()
