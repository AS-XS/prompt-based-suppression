from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_split import build  # noqa: E402
from src.data.wmdp import EXPECTED_N, SUBSETS, load_wmdp, wmdp_item  # noqa: E402


def _rows(n, prefix):
    return [{"question": f"{prefix} q{i}?", "choices": ["a", "b", "c", "d"], "answer": i % 4}
            for i in range(n)]


def _items(n_bio=1273, n_cyber=1987):
    return ([wmdp_item(r, "wmdp-bio") for r in _rows(n_bio, "bio")]
            + [wmdp_item(r, "wmdp-cyber") for r in _rows(n_cyber, "cy")])


def test_wmdp_items_carry_the_letter_answer_not_the_index():


    it = wmdp_item({"question": "q?", "choices": ["w", "x", "y", "z"], "answer": 2}, "wmdp-bio")
    assert it.answer_key == "C"
    assert it.reference == "y"
    assert it.meta["choices"] == ["w", "x", "y", "z"]
    assert it.meta["format"] == "mc"          


def test_item_ids_are_unique_and_carry_the_choices():


    a = wmdp_item({"question": "same?", "choices": ["1", "2", "3", "4"], "answer": 0}, "wmdp-bio")
    b = wmdp_item({"question": "same?", "choices": ["5", "6", "7", "8"], "answer": 0}, "wmdp-bio")
    assert a.item_id != b.item_id
    ids = {i.item_id for i in _items(50, 50)}
    assert len(ids) == 100


def test_subtype_is_hazard_so_the_isolated_scorers_skip_it():


    it = wmdp_item({"question": "q?", "choices": ["a", "b", "c", "d"], "answer": 0}, "wmdp-cyber")
    assert it.concept_subtype == "hazard"
    assert it.concept_class == "general"
    assert it.probe_type == "forget"


def test_unknown_subset_is_rejected_by_name():
    with pytest.raises(KeyError, match="wmdp-bio"):
        load_wmdp("wmdp-biology")            


def test_expected_sizes_are_recorded_for_every_subset():
    assert set(EXPECTED_N) == set(SUBSETS)
    assert EXPECTED_N["wmdp-bio"] == 1273 and EXPECTED_N["wmdp-cyber"] == 1987


def test_split_is_deterministic_and_per_concept():
    items = _items()
    a = build(items, 0.25, seed=0)
    assert a == build(items, 0.25, seed=0), "same seed must give the same split"
    assert set(a) == {"wmdp-bio", "wmdp-cyber"}, "each subset is split independently"
    assert len(a["wmdp-bio"]) == 318 and len(a["wmdp-cyber"]) == 497
    assert not (set(a["wmdp-bio"]) & set(a["wmdp-cyber"]))
    assert build(items, 0.25, seed=1) != a, "a different seed must give a different split"


def test_split_does_not_depend_on_the_order_items_were_written():


    items = _items(200, 200)
    assert build(items, 0.25, 0) == build(list(reversed(items)), 0.25, 0)


def test_only_forget_items_are_split():

    from src.data.mmlu import mc_item
    retain = mc_item({"question": "r?", "choices": ["a", "b", "c", "d"], "answer": 1},
                     "biology", "college_biology", probe_type="retain")
    out = build(_items(40, 40) + [retain], 0.25, 0)
    assert "biology" not in out


def _run(tmp_path, *args):
    ip, sp = tmp_path / "items.jsonl", tmp_path / "split.json"
    if not ip.exists():
        ip.write_text("\n".join(i.model_dump_json() for i in _items(200, 200)), encoding="utf-8")
    return subprocess.run([sys.executable, "scripts/build_split.py", "--items", str(ip),
                           "--out", str(sp), *args], cwd=ROOT, capture_output=True, text=True), sp


def test_an_existing_split_is_never_silently_overwritten(tmp_path):


    r, sp = _run(tmp_path)
    assert r.returncode == 0, r.stderr
    before = json.loads(sp.read_text())
    r2, _ = _run(tmp_path)
    assert json.loads(sp.read_text()) == before
    assert "--force" in r2.stdout


def test_a_stale_split_FAILS_rather_than_filtering_to_survivors(tmp_path):


    r, sp = _run(tmp_path)
    assert r.returncode == 0
    (tmp_path / "items.jsonl").write_text(
        "\n".join(i.model_dump_json() for i in _items(10, 10)), encoding="utf-8")
    r2, _ = _run(tmp_path, "--check")
    assert r2.returncode == 1
    assert "STALE" in r2.stdout


def test_the_split_file_says_it_is_ours_and_must_be_cited(tmp_path):


    r, sp = _run(tmp_path)
    payload = json.loads(sp.read_text())
    assert "OURS" in payload["note"] and "test" in payload["note"]
    assert payload["seed"] == 0 and payload["val_frac"] == 0.25


def _scored(item_id, probe_type="forget"):
    return {"item_id": item_id, "probe_type": probe_type}


def test_no_split_argument_changes_nothing(tmp_path):

    from src.data.split import select

    rows = [_scored("a"), _scored("b"), _scored("c", "retain")]
    assert select(rows, {"a"}, None) == rows


def test_val_and_test_partition_the_forget_arm(tmp_path):
    from src.data.split import select

    rows = [_scored(x) for x in "abcd"]
    val_ids = {"a", "c"}
    v = [r["item_id"] for r in select(rows, val_ids, "val")]
    t = [r["item_id"] for r in select(rows, val_ids, "test")]
    assert v == ["a", "c"] and t == ["b", "d"]
    assert not set(v) & set(t) and sorted(v + t) == list("abcd")


def test_retain_arms_pass_through_both_halves():


    from src.data.split import select

    rows = [_scored("n1", "neighbor"), _scored("u1", "utility"), _scored("r1", "retain")]
    for which in ("val", "test"):
        assert len(select(rows, set(), which)) == 3


def test_a_missing_split_file_is_an_error_not_an_empty_set(tmp_path):


    from src.data.split import read_val_ids

    with pytest.raises(SystemExit):
        read_val_ids(tmp_path / "nope.json")


def test_score_accuracy_split_writes_its_own_directory(tmp_path):

    items = _items(40, 40)
    ip, sp = tmp_path / "items.jsonl", tmp_path / "split.json"
    ip.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")
    subprocess.run([sys.executable, "scripts/build_split.py", "--items", str(ip), "--out", str(sp)],
                   cwd=ROOT, capture_output=True, text=True, check=True)
    gens = [{"model": "M", "prompt_id": "none-baseline", "condition": "baseline",
             "think_state": "off", "concept_id": i.concept_id, "item_id": i.item_id,
             "raw_output": "ANSWER: B"} for i in items]
    gp = tmp_path / "gen.jsonl"
    gp.write_text("\n".join(json.dumps(g) for g in gens) + "\n", encoding="utf-8")

    seen = {}
    for which in ("val", "test"):
        out = tmp_path / "analysis" / which
        r = subprocess.run([sys.executable, "scripts/score_accuracy.py", "--items", str(ip),
                            "--generations", str(gp), "--out-dir", str(out),
                            "--split", which, "--split-file", str(sp)],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        seen[which] = (out / "accuracy.csv").read_text()
    assert seen["val"] != seen["test"]
    assert (tmp_path / "analysis" / "val" / "accuracy.csv").exists()
