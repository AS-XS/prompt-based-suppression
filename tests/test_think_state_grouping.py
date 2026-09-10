from __future__ import annotations

from src.analysis.accuracy import accuracy_table, delta_table


def _r(ts, pid, item, correct, subject="s"):
    return {"model": "m", "think_state": ts, "mmlu_subject": subject, "prompt_id": pid,
            "probe_type": "forget", "item_id": item, "parsed": True, "correct": correct}


def test_states_are_not_pooled():

    scored = [_r("off", "none-baseline", "i1", True), _r("on", "none-baseline", "i1", False)]
    rows = accuracy_table(scored)
    by_ts = {r["think_state"]: r for r in rows}
    assert set(by_ts) == {"off", "on"}
    assert by_ts["off"]["accuracy"] == 1.0 and by_ts["on"]["accuracy"] == 0.0


def test_delta_does_not_overwrite_one_state_with_the_other():

    scored = [
        _r("off", "none-baseline", "i1", True), _r("off", "world-frame", "i1", True),   
        _r("on", "none-baseline", "i1", True), _r("on", "world-frame", "i1", False),    
    ]
    rows, supp = delta_table(scored)
    assert supp == ["world-frame"]
    by_ts = {r["think_state"]: r for r in rows}
    assert len(rows) == 2, "one row per think-state; got a collision"
    assert by_ts["off"]["world-frame_drop"] == 0.0
    assert by_ts["on"]["world-frame_drop"] == 1.0


def test_each_state_pairs_on_its_own_items():
    scored = [_r("off", "none-baseline", "i1", True), _r("off", "world-frame", "i1", False),
              _r("on", "none-baseline", "i2", True)]   
    rows, _ = delta_table(scored)
    by_ts = {r["think_state"]: r for r in rows}
    assert by_ts["off"]["world-frame_n"] == 1      
    assert by_ts["on"]["world-frame_n"] == 0       
