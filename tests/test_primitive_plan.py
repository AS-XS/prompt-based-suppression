import csv
import json

import pytest

from src.data.schema import GenerationRow, Sampling
from src.gen.runner import append_jsonl
from scripts import judge_primitives as judge


@pytest.mark.parametrize("shard", [None, (1, 2), (2, 2)])
def test_filtered_sharded_work_matches_execution_and_resume(tmp_path, monkeypatch, shard):
    rows = [GenerationRow(git_commit="c", config_hash="h", config_id="c", model="M",
                          item_id=f"i{i}", concept_id="physics", concept_class="general",
                          concept_subtype="subject", probe_type="forget" if i % 2 else "utility",
                          prompt_id="no-meta", condition="suppressed" if i else "baseline",
                          think_state="on", sampling=Sampling(),
                          raw_output="<think>reasoning</think>Answer: A\nWhy: example.")
            for i in range(8)]
    gens, out = tmp_path / "gens.jsonl", tmp_path / "judged"
    append_jsonl(gens, rows)
    calls = []

    def fake(llm, sp, **kwargs):
        calls.append(kwargs)
        return {"surface": kwargs["surface"], "raw_judge_output": "verbatim",
                "semantic_leak": False, "parsed_ok": True}

    monkeypatch.setattr(judge, "run_primitive_judge", fake)
    args = (gens, out, True, 3, ("explanation", "trace"), None, shard, ("forget",))
    expected = 6 if shard is None else (4 if shard == (1, 2) else 2)
    assert judge.outstanding(*args) == expected
    assert not out.exists(), "preflight should not create output directories"
    n, path = judge._judge_file(None, None, gens, out, "J", *args[2:])
    assert n == expected == len(calls)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["probe_type"] for r in records} == {"forget"}
    assert {r["surface"] for r in records} == {"trace", "explanation"}
    with path.with_suffix(".csv").open(newline="") as f:
        assert len(list(csv.DictReader(f))) == expected
    assert judge.outstanding(*args) == 0
    judge._judge_file(None, None, gens, out, "J", *args[2:])
    assert len(calls) == expected
