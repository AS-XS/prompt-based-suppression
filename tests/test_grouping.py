from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.accuracy import accuracy_table, score_generations  # noqa: E402
from src.analysis.grouping import eval_subject  # noqa: E402
from src.data.mmlu import mc_item  # noqa: E402
from src.data.wmdp import wmdp_item  # noqa: E402

_ROW = {"question": "q?", "choices": ["a", "b", "c", "d"], "answer": 1}


def _as_dict(item) -> dict:
    return json.loads(item.model_dump_json())


def test_wmdp_forget_item_groups_by_its_subset():
    for subset in ("wmdp-bio", "wmdp-cyber"):
        assert eval_subject(_as_dict(wmdp_item(_ROW, subset))) == subset


def test_bio_and_cyber_do_not_share_a_row():
    items = {}
    gens = []
    for subset in ("wmdp-bio", "wmdp-cyber"):
        for i in range(3):
            it = wmdp_item({**_ROW, "question": f"{subset} q{i}?"}, subset)
            items[it.item_id] = {"answer_key": it.answer_key, "format": "mc",
                                 "mmlu_subject": eval_subject(_as_dict(it)),
                                 "probe_type": "forget", "concept_id": subset}
            gens.append({"model": "M", "prompt_id": "none-baseline", "condition": "baseline",
                         "think_state": "off", "concept_id": subset, "item_id": it.item_id,
                         "raw_output": "ANSWER: B"})
    rows = accuracy_table(score_generations(gens, items))
    subjects = sorted(r["mmlu_subject"] for r in rows)
    assert subjects == ["wmdp-bio", "wmdp-cyber"], subjects
    assert all(r["n"] == 3 for r in rows)          


def test_mmlu_item_keeps_its_own_subject():

    for probe in ("forget", "retain", "utility", "neighbor"):
        it = _as_dict(mc_item(_ROW, "biology", "high_school_biology", probe_type=probe))
        assert eval_subject(it) == "high_school_biology"


def test_retain_item_without_a_subject_is_none_not_the_suppressed_concept():


    for probe in ("retain", "neighbor", "utility"):
        assert eval_subject({"concept_id": "wmdp-bio", "probe_type": probe, "meta": {}}) is None


def _write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_score_accuracy_splits_subsets_and_keeps_the_neighbor_arm(tmp_path):


    items, gens = [], []
    for subset in ("wmdp-bio", "wmdp-cyber"):
        built = [wmdp_item({**_ROW, "question": f"{subset} q{i}?"}, subset) for i in range(4)]
        built += [mc_item({**_ROW, "question": f"{subset} n{i}?"}, subset, "college_biology",
                          probe_type="neighbor") for i in range(4)]
        for it in built:
            items.append(json.loads(it.model_dump_json()))
            for prompt, out in (("none-baseline", "ANSWER: B"), ("scoped-ignorance", "ANSWER: C")):
                gens.append({"model": "M", "prompt_id": prompt, "think_state": "off",
                             "condition": "baseline" if prompt == "none-baseline" else "suppressed",
                             "concept_id": subset, "item_id": it.item_id, "raw_output": out})
    _write(tmp_path / "items.jsonl", items)
    _write(tmp_path / "gen.jsonl", gens)

    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "score_accuracy.py"),
                          "--items", str(tmp_path / "items.jsonl"),
                          "--generations", str(tmp_path / "gen.jsonl"),
                          "--out-dir", str(tmp_path / "analysis")],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr

    def read(name):
        return list(csv.DictReader((tmp_path / "analysis" / name).open(encoding="utf-8")))

    forget = {r["mmlu_subject"] for r in read("accuracy.csv") if r["probe_type"] == "forget"}
    assert forget == {"wmdp-bio", "wmdp-cyber"}, forget
    assert {r["mmlu_subject"] for r in read("accuracy_delta.csv")} == {"wmdp-bio", "wmdp-cyber"}

    side = read("side_effect.csv")
    assert side, "the neighbor arm vanished from side_effect.csv"
    assert {(r["concept_id"], r["mmlu_subject"]) for r in side} == {
        ("wmdp-bio", "college_biology"), ("wmdp-cyber", "college_biology")}


def test_mmlu_subjects_still_fold_to_their_concept():
    from src.analysis.grouping import tested_concept
    from src.data.mmlu import SUBJECT_TO_CONCEPT

    row = {"mmlu_subject": "college_biology", "concept_id": "physics"}
    assert tested_concept(row, SUBJECT_TO_CONCEPT) == "biology"


def test_a_wmdp_forget_row_folds_to_its_own_subset():
    from src.analysis.grouping import tested_concept
    from src.data.mmlu import SUBJECT_TO_CONCEPT

    for c in ("wmdp-bio", "wmdp-cyber"):
        assert tested_concept({"mmlu_subject": c, "concept_id": c}, SUBJECT_TO_CONCEPT) == c


def test_an_unmapped_subject_that_is_not_the_suppressed_concept_is_still_dropped():


    from src.analysis.grouping import tested_concept
    from src.data.mmlu import SUBJECT_TO_CONCEPT

    assert tested_concept({"mmlu_subject": "marketing", "concept_id": "biology"},
                          SUBJECT_TO_CONCEPT) is None


def test_no_real_mmlu_subject_can_reach_the_new_branch():


    from src.data.mmlu import CONCEPT_SUBJECTS, SUBJECT_TO_CONCEPT
    from src.data.retain import FAR_RETAIN, STEM_SUBJECT

    universe = {s for v in CONCEPT_SUBJECTS.values() for s in v} | set(STEM_SUBJECT.values())
    universe |= {s for v in FAR_RETAIN.values() for s in (v if isinstance(v, (list, tuple)) else [v])}
    assert not (universe & set(CONCEPT_SUBJECTS)), "an MMLU subject now equals a concept id"
    assert all(s in SUBJECT_TO_CONCEPT for s in {s for v in CONCEPT_SUBJECTS.values() for s in v})


def test_wmdp_neighbours_fold_to_their_own_concept_not_to_mmlu():


    from src.analysis.grouping import tested_concept
    from src.data.mmlu import SUBJECT_TO_CONCEPT
    from src.data.wmdp import NEAR_RETAIN

    for concept, subjects in NEAR_RETAIN.items():
        for subject in subjects:
            row = {"mmlu_subject": subject, "concept_id": concept}
            assert tested_concept(row, SUBJECT_TO_CONCEPT) == concept, (concept, subject)

    assert tested_concept({"mmlu_subject": "wmdp-bio", "concept_id": "wmdp-bio"},
                          SUBJECT_TO_CONCEPT) == "wmdp-bio"


def test_the_mmlu_series_is_untouched_by_the_wmdp_neighbour_fold():


    from src.analysis.grouping import tested_concept
    from src.data.mmlu import SUBJECT_TO_CONCEPT
    assert tested_concept({"mmlu_subject": "college_biology", "concept_id": "biology"},
                          SUBJECT_TO_CONCEPT) == "biology"
    assert tested_concept({"mmlu_subject": "college_biology", "concept_id": "physics"},
                          SUBJECT_TO_CONCEPT) == "biology"

    assert tested_concept({"mmlu_subject": "marketing", "concept_id": "wmdp-bio"},
                          SUBJECT_TO_CONCEPT) is None


def test_the_pooled_caption_counts_the_concepts_it_actually_describes():


    ns = {"__name__": "pt", "__file__": "scripts/present_tables.py"}
    src = open("scripts/present_tables.py").read().split("def main(")[0]
    exec(compile(src, "pt", "exec"), ns)
    mmlu = [{"probe_type": "forget", "concept_id": c} for c in
            ("biology", "physics", "chemistry", "mathematics", "statistics", "computer_science")]
    wmdp = [{"probe_type": "forget", "concept_id": c} for c in ("wmdp-bio", "wmdp-cyber")]
    assert "6 suppressed concepts" in ns["_note"](mmlu)
    assert "mathematics" in ns["_note"](mmlu)          
    assert "2 suppressed concepts" in ns["_note"](wmdp)
    assert "mathematics" not in ns["_note"](wmdp)      


def test_coverage_allows_fewshot_waves_to_be_short_by_their_held_out_examples():


    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "cov", Path(__file__).resolve().parents[1] / "scripts" / "coverage.py")
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)
    assert cov._is_fewshot("scoped-directed-fewshot-grouped@specific")
    assert cov._is_fewshot("scoped-directed-fewshot@specific")
    assert not cov._is_fewshot("scoped-directed-aware@specific")


    assert 0 < cov._FEWSHOT_SLACK < 1000
