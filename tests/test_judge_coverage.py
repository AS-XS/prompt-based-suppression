from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from judge_coverage import coverage  # noqa: E402

FP = {"concept": "cccccccccccccccc", "framing": "ffffffffffffffff",
      "mention": "mmmmmmmmmmmmmmmm"}


def _row(gid, surface="trace", *, concept=None, framing=None, mention=None,
         concept_hash=FP["concept"], framing_hash=FP["framing"], mention_hash=FP["mention"]):
    r = {"generation_id": gid, "surface": surface}
    if concept is not None:
        r["concept_score"] = concept
        r["judge_config_hash"] = concept_hash      
    if framing is not None:
        r["framing_score"] = framing
        r["framing_config_hash"] = framing_hash
    if mention is not None:
        r["mention_score"] = mention
        r["mention_config_hash"] = mention_hash
    return r


def test_a_complete_sample_reads_complete():
    rows = [_row("g1", concept=2, framing=0, mention=2),
            _row("g2", concept=2, framing=2, mention=0)]
    cov = coverage(rows, FP)
    for stage in ("concept", "framing", "mention"):
        n, d, other = cov[stage]
        assert (n, d) == (2, 2), (stage, n, d)
        assert other == {}


def test_framing_denominator_is_every_row_as_the_judge_asks_it():


    rows = [_row("g1", concept=2, framing=0, mention=2),
            _row("g2", concept=0, mention=0)]            
    cov = coverage(rows, FP)
    assert cov["framing"][:2] == (1, 2), cov["framing"]
    assert cov["concept"][:2] == (2, 2)
    assert cov["mention"][:2] == (2, 2)


def test_a_superseded_rubric_does_not_count_as_covered():


    rows = [_row("g1", concept=2, framing=2, framing_hash="0000old0000old00"),
            _row("g2", concept=2, framing=0, framing_hash="0000old0000old00")]
    cov = coverage(rows, FP)
    n, d, other = cov["framing"]
    assert (n, d) == (0, 2), "scored, but not at the rubric the code computes now"
    assert other == {"0000old0000old00": 2}


def test_the_newest_record_wins_for_a_pair():


    rows = [_row("g1", concept=2, framing=2, framing_hash="0000old0000old00"),
            _row("g1", concept=2, framing=0)]            
    n, d, other = coverage(rows, FP)["framing"]
    assert (n, d) == (1, 1)
    assert other == {"0000old0000old00": 1}


def test_surfaces_are_counted_separately():


    rows = [_row("g1", "trace", concept=2, framing=0, mention=2),
            _row("g1", "explanation", concept=2, mention=2)]   
    cov = coverage(rows, FP)
    assert cov["concept"][:2] == (2, 2)
    assert cov["framing"][:2] == (1, 2), "the explanation surface is still owed a framing score"


def test_a_stage_never_run_reads_zero_not_complete():
    rows = [_row("g1", concept=2), _row("g2", concept=2)]
    cov = coverage(rows, FP)
    assert cov["concept"][:2] == (2, 2)
    assert cov["framing"][:2] == (0, 2)
    assert cov["mention"][:2] == (0, 2)
