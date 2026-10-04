from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.agreement import (compare, judge_labels, parse_grade_sheet,  # noqa: E402
                                    parse_labels)


def _sheet(*rows):
    return [{"n": i, "present": p, "used": u, "named": m, "notes": ""}
            for i, (p, u, m) in enumerate(rows, 1)]


def test_a_sheet_is_read_by_the_stage_questions_and_blank_rows_wait():
    got = parse_labels(_sheet(("2", "y", "2"), ("1", "N", "0"), ("0", "", "1"), ("", "", "")))
    assert got == {1: {"present": 2, "used": True, "named": 2},
                   2: {"present": 1, "used": False, "named": 0},
                   3: {"present": 0, "used": False, "named": 1}}       


@pytest.mark.parametrize("row,why", [
    (("3", "y", "0"), "0, 1 or 2"),            
    (("2", "", "1"), "used must be y or n"),   
    (("0", "y", "0"), "present is 0"),         
])
def test_a_typo_is_refused_not_scored_as_a_disagreement(row, why):
    with pytest.raises(ValueError, match=why):
        parse_labels(_sheet(row))


def test_the_judge_reads_its_key_with_the_same_rules_as_the_tables():


    key = [{"n": 1, "concept_score": "2", "framing_score": "0", "mention_score": "2"},
           {"n": 2, "concept_score": "2", "framing_score": "2", "mention_score": "0"},
           {"n": 3, "concept_score": "0", "framing_score": "", "mention_score": "1"},
           {"n": 4, "concept_score": "1", "framing_score": "", "mention_score": "0"},
           {"n": 5, "concept_score": "", "framing_score": "", "mention_score": "0"}]
    assert judge_labels(key) == {1: {"present": 2, "used": True, "named": 2},
                                 2: {"present": 2, "used": False, "named": 0},
                                 3: {"present": 0, "used": False, "named": 1}}


def test_the_comparison_reports_kappa_beside_its_2x2():
    a = {1: {"present": 2, "used": True, "named": 2}, 2: {"present": 1, "used": True, "named": 0},
         3: {"present": 0, "used": False, "named": 0}, 4: {"present": 2, "used": False, "named": 2}}
    b = {1: {"present": 2, "used": True, "named": 1}, 2: {"present": 2, "used": False, "named": 0},
         3: {"present": 0, "used": False, "named": 0}, 9: {"present": 2, "used": True, "named": 2}}
    rows = {r["axis"]: r for r in compare(a, b)}
    assert rows["present"]["n"] == 3                         
    assert (rows["present"]["agreement"], rows["present"]["kappa"]) == (1.0, 1.0)
    assert rows["present"]["exact_0_1_2"] == round(2 / 3, 4)  
    u = rows["used"]
    assert (u["both"], u["a_only"], u["b_only"], u["neither"]) == (1, 1, 0, 1)
    assert u["exact_0_1_2"] is None                          
    assert rows["named"]["agreement"] == 1.0 and rows["named"]["exact_0_1_2"] == round(2 / 3, 4)


def _sheet_text(*grades):
    return "# header\n" + "".join(f"\n## {i}\n<!-- row g{i} -->\npassage\nGRADE {g}\n"
                                   for i, g in enumerate(grades, 1))


def test_the_grading_sheet_is_read_on_the_judges_own_scales():


    ids, got = parse_grade_sheet(_sheet_text("concept=2 framing=0 mention=2",
                                             "concept=  framing=  mention=",
                                             "concept = 0 framing= mention =1",
                                             "concept=1 framing=2 mention=0"))
    assert ids == {1: "g1", 2: "g2", 3: "g3", 4: "g4"}
    assert set(got) == {1, 3, 4}
    assert (got[1]["used"], got[3]["used"], got[4]["used"]) == (True, False, False)
    assert (got[3]["framing"], got[4]["present"], got[4]["named"]) == (None, 1, 0)


@pytest.mark.parametrize("grade,why", [
    ("concept=0 framing=0 mention=0", "leave framing empty"),
    ("concept=2 framing= mention=2", "framing must be 0"),
    ("concept=2 framing=1 mention=2", "framing must be 0"),      
    ("concept=3 framing=0 mention=2", "0, 1 or 2"),
])
def test_a_malformed_grade_is_refused_naming_its_entry(grade, why):
    with pytest.raises(ValueError, match=f"entry 1: .*{why}"):
        parse_grade_sheet(_sheet_text(grade))

