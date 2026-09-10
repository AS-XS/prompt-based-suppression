from __future__ import annotations

from src.analysis.recall import (exact_match, is_hit, recall_delta, recall_table, score_recall,
                                  token_f1, token_recall)


def test_token_recall():
    assert token_recall("Mitochondria produce ATP energy.", "mitochondria") == 1.0
    assert token_recall("I don't know who that is.", "Stephen King") == 0.0
    assert token_recall("The powerhouse of the cell", "powerhouse cell") == 1.0


def test_is_hit_substring_and_threshold():
    assert is_hit("The author is Stephen King, famously.", "Stephen King") is True
    assert is_hit("I cannot help with that.", "Stephen King") is False


def _idx(**probe_types):

    return {iid: {"reference": ref, "probe_type": pt, "concept_id": "king"}
            for iid, (ref, pt) in probe_types.items()}


def test_score_and_recall_table():
    idx = _idx(i1=("Stephen King", "forget"), i2=("Maine", "neighbor"))
    gens = [
        {"item_id": "i1", "model": "m", "prompt_id": "none-baseline", "raw_output": "Stephen King wrote it."},
        {"item_id": "i2", "model": "m", "prompt_id": "none-baseline", "raw_output": "It is set in Maine."},
    ]
    scored = score_recall(gens, idx)
    assert {s["probe_type"] for s in scored} == {"forget", "neighbor"}
    rows = recall_table(scored, by=("probe_type",))
    by_pt = {r["probe_type"]: r for r in rows}
    assert by_pt["forget"]["hit_rate"] == 1.0 and by_pt["neighbor"]["hit_rate"] == 1.0


def test_token_f1_and_exact_match():


    assert exact_match("Stephen King", "stephen king") is True
    assert token_f1("Stephen King", "stephen king") == 1.0
    assert exact_match("The Beatles!", "beatles") is True           
    verbose = "I think the answer is Stephen King, the horror writer."
    assert token_recall(verbose, "Stephen King") == 1.0            
    assert 0.0 < token_f1(verbose, "Stephen King") < token_recall(verbose, "Stephen King")
    assert token_f1("I don't know.", "Stephen King") == 0.0 and exact_match("I don't know.", "x") is False


def test_recall_table_now_reports_f1_and_em():
    idx = _idx(i1=("Stephen King", "forget"))
    gens = [{"item_id": "i1", "model": "m", "prompt_id": "none-baseline", "raw_output": "Stephen King"}]
    (row,) = recall_table(score_recall(gens, idx), by=("probe_type",))
    assert row["em_rate"] == 1.0 and row["mean_f1"] == 1.0


def test_recall_delta_supports_f1_and_em_metrics():
    scored = [
        {"model": "m", "concept_id": "king", "prompt_id": "none-baseline", "item_id": "i1",
         "recall": 1.0, "hit": True, "f1": 1.0, "em": True},
        {"model": "m", "concept_id": "king", "prompt_id": "world-frame", "item_id": "i1",
         "recall": 0.0, "hit": False, "f1": 0.0, "em": False},
    ]
    for metric in ("f1", "em"):
        rows, _ = recall_delta(scored, by=("model", "concept_id"), metric=metric)
        assert rows[0]["world-frame_drop"] == 1.0, metric


def test_recall_delta_paired_drop():

    scored = [
        {"model": "m", "concept_id": "king", "prompt_id": "none-baseline", "item_id": "i1",
         "recall": 1.0, "hit": True},
        {"model": "m", "concept_id": "king", "prompt_id": "world-frame", "item_id": "i1",
         "recall": 0.0, "hit": False},
    ]
    rows, supp = recall_delta(scored, by=("model", "concept_id"))
    assert supp == ["world-frame"]
    (row,) = rows
    assert row["baseline"] == 1.0 and row["world-frame"] == 0.0 and row["world-frame_drop"] == 1.0
