from __future__ import annotations

import importlib.util
from pathlib import Path

from src.analysis.recall import recall_table

_SPEC = importlib.util.spec_from_file_location(
    "present_tables", Path(__file__).resolve().parents[1] / "scripts" / "present_tables.py")
present_tables = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(present_tables)


def _gen(item_id, prompt_id, raw, condition="suppressed"):
    return {"item_id": item_id, "model": "M", "prompt_id": prompt_id, "condition": condition,
            "think_state": "off", "concept_id": "Stephen King", "raw_output": raw}


def test_isolated_table_ignores_mmlu_rows_even_though_they_carry_a_reference():

    idx = {
        "ent-1": {"reference": "Bangor, Maine", "probe_type": "forget",
                  "concept_subtype": "entity", "concept_id": "Stephen King"},

        "mc-1": {"reference": "directional selection", "probe_type": "retain",
                 "concept_subtype": "subject", "concept_id": "biology"},
    }
    gens = [_gen("ent-1", "none-baseline", "Bangor, Maine.", condition="baseline"),
            _gen("ent-1", "world-frame", "I could not say."),
            _gen("mc-1", "none-baseline", "Answer: A", condition="baseline"),
            _gen("mc-1", "world-frame", "Answer: A")]

    rows = present_tables._isolated_rows(gens, idx)

    probes = {r["probe_type"] for r in rows}
    assert probes == {"forget"}, f"MMLU rows leaked into the isolated table: {probes}"


def test_isolated_table_defaults_to_recall_not_f1():


    idx = {"ent-1": {"reference": "Bangor, Maine", "probe_type": "forget",
                     "concept_subtype": "entity", "concept_id": "Stephen King"}}
    verbose_but_right = "He was born in Bangor, Maine, a small city in the northeast, back in 1947."
    gens = [_gen("ent-1", "none-baseline", verbose_but_right, condition="baseline"),
            _gen("ent-1", "world-frame", verbose_but_right)]

    (row,) = present_tables._isolated_rows(gens, idx)

    assert row["world-frame_drop"] == 0.0
    assert row["baseline_acc"] == 1.0, (
        f"default metric is length-sensitive (got {row['baseline_acc']}); recall gives 1.0 here")


def test_recall_csv_columns_are_derived_so_a_new_metric_cannot_be_dropped():

    scored = [{"model": "M", "think_state": "off", "prompt_id": "world-frame",
               "probe_type": "forget", "recall": 0.5, "hit": True, "f1": 0.4, "em": False}]
    produced = set(recall_table(scored)[0])
    assert {"mean_f1", "em_rate"} <= produced, "recall_table stopped emitting the T1 metrics"

    src = (Path(__file__).resolve().parents[1] / "scripts" / "score_recall.py").read_text()
    assert '"mean_recall", "hit_rate"]' not in src, (
        "score_recall.py is back to a hardcoded recall.csv fieldname list — mean_f1/em_rate will be "
        "silently dropped by extrasaction='ignore' again")
