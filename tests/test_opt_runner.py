import json
import sys
from types import SimpleNamespace

import pytest

dspy = pytest.importorskip("dspy")


@pytest.fixture
def search_run(tmp_path, monkeypatch):
    from dspy.utils.dummies import DummyLM
    from src.opt import preparation
    from scripts import optimize_prompt

    items = []
    for tier in ("forget", "neighbor", "utility"):
        for i in range(4):
            items.append(dict(item_id=f"{tier}-{i}", concept_id="physics",
                              concept_class="general", concept_subtype="subject",
                              probe_type=tier, question=f"Example {tier} {i}?", answer_key="B",
                              meta={"format": "mc", "choices": ["x", "y", "z", "w"]}))
    item_path, split_path = tmp_path / "items.jsonl", tmp_path / "split.json"
    item_path.write_text("\n".join(json.dumps(x) for x in items))
    split_path.write_text(json.dumps({"val": {"physics": [x["item_id"] for x in items]}}))
    baseline = tmp_path / "Meta-Llama-3-8B-Instruct__none-baseline.jsonl"
    baseline.write_text("\n".join(json.dumps(dict(item_id=x["item_id"], think_state="na",
                                                condition="baseline", raw_output="Answer: B"))
                                  for x in items))
    monkeypatch.setattr(preparation, "items_path", lambda run: item_path)
    monkeypatch.setattr(preparation, "split_path", lambda run: split_path)
    monkeypatch.setattr(preparation, "generations_dir", lambda run: tmp_path)
    monkeypatch.setattr(dspy, "LM", lambda *a, **k: DummyLM(
        [{"response": "Answer: A\nWhy: unknown."}] * 200))

    calls = []

    class Search:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def compile(self, program, **kwargs):
            calls.append(kwargs)
            found = program.deepcopy()
            found.signature = found.signature.with_instructions("Set aside physics.")
            found.demos = kwargs["trainset"][:kwargs["max_labeled_demos"]]
            found.candidate_programs = [{"program": found.deepcopy(), "score": 100.0}]
            found.trial_logs = {0: {"score": 100.0}}
            return found

    monkeypatch.setattr(dspy.teleprompt, "MIPROv2", Search)
    out = tmp_path / "result.json"

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["optimize_prompt.py", "--run", "exp1", "--model",
                            "Meta-Llama-3-8B-Instruct", "--think-state", "na", "--concept",
                            "physics", "--seed-prompt", "scoped-world", "--out", str(out),
                            *extra])
        return optimize_prompt.main()

    return SimpleNamespace(run=run, out=out, calls=calls)


@pytest.mark.parametrize("flags", [("--dry-run",), (), ("--demos", "2"),
                                  ("--prompt-model", "Meta-Llama-3-8B-Instruct")])
def test_cli_banks_and_completes_search(search_run, flags):
    assert search_run.run(*flags) == 0
    result = json.loads(search_run.out.read_text())
    assert result["status"] == "complete"
    assert result["n_examples"] == 12
    assert result["n_dev"] == len(result["seed_items"]) == len(result["found_items"])
    assert result["arms_requested_but_empty"] == []
    assert result["dev_mix"].keys() == {"forget", "neighbor", "utility"}
    assert result["found_score"] == result["seed_score"]
    if "--dry-run" in flags:
        assert search_run.calls == []
        assert result["search"] == {}
    else:
        assert search_run.calls[0]["auto"] is None
        assert search_run.calls[1]["num_trials"] == 18
        assert search_run.calls[1]["minibatch"] is False
        assert result["search"]["n_candidates"] == 1
        assert result["search"]["candidates"][0]["n_demos"] == result["demos"]


def test_incompatible_result_survives_seed_banking(search_run):
    original = '{"status":"complete","model":"different-model"}\n'
    search_run.out.write_text(original)
    with pytest.raises(SystemExit, match="DIFFERENT experiment"):
        search_run.run()
    assert search_run.out.read_text() == original
    assert search_run.calls == []
    assert search_run.run("--force") == 0


def test_compile_failure_leaves_seed_progress(search_run, monkeypatch):
    from src.opt import runner

    def fail(*args):
        raise RuntimeError("search interrupted")

    monkeypatch.setattr(runner, "compile_search", fail)
    with pytest.raises(RuntimeError, match="search interrupted"):
        search_run.run()
    result = json.loads(search_run.out.read_text())
    assert result["status"].startswith("seed-only")
    assert result["seed_items"]
    assert "found_score" not in result
