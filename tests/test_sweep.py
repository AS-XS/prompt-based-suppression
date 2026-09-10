from __future__ import annotations

from src.data.schema import GenerationRow, Item, Sampling, make_item_id
from src.gen.runner import read_jsonl
from src.gen.sweep import run_sweep
from src.gen.thinkstate import chat_kwargs, think_states_for


def test_think_states_per_family():
    assert think_states_for("Qwen3-8B") == ["off", "on"]
    assert think_states_for("DeepSeek-R1-Distill-Llama-8B") == ["on"]
    assert think_states_for("GLM-Z1-9B-0414") == ["on"]
    assert think_states_for("Llama-3.1-8B-Instruct") == ["na"]
    assert think_states_for("Qwen2.5-7B-Instruct") == ["na"]  


def test_qwen3_toggle_kwargs():
    assert chat_kwargs("Qwen3-8B", "on") == {"chat_template_kwargs": {"enable_thinking": True}}
    assert chat_kwargs("Qwen3-8B", "off") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert chat_kwargs("Llama-3.1-8B-Instruct", "na") == {}


class _Out:
    def __init__(self, text: str):
        self.outputs = [type("O", (), {"text": text, "token_ids": [1, 2, 3],
                                        "finish_reason": "stop"})()]
        self.prompt_token_ids = [0, 0]


class _FakeLLM:
    def chat(self, messages, sp, **kw):
        return [_Out(f"answer {i}") for i in range(len(messages))]


def _item() -> Item:
    q = "What is a cell?"
    return Item(item_id=make_item_id("biology", "forget", q), concept_id="biology",
                concept_class="general", concept_subtype="subject", probe_type="forget", question=q)


def test_run_sweep_counts_and_resumes(tmp_path):
    out = tmp_path / "g.jsonl"
    s = Sampling(temperature=0.0, seed=0)
    kw = dict(conditions=["baseline", "suppressed"], suppressed_prompt="world-frame", sampling=s)


    n1 = run_sweep("Llama-3.1-8B-Instruct", "/fake", [_item()], out, llm=_FakeLLM(), **kw)
    assert n1 == 2
    rows = list(read_jsonl(out, GenerationRow))
    assert {r.condition for r in rows} == {"baseline", "suppressed"}
    assert all(r.raw_output.startswith("answer") for r in rows)


    assert run_sweep("Llama-3.1-8B-Instruct", "/fake", [_item()], out, llm=_FakeLLM(), **kw) == 0


def test_qwen3_yields_both_think_states(tmp_path):
    out = tmp_path / "q.jsonl"
    n = run_sweep("Qwen3-8B", "/fake", [_item()], out, llm=_FakeLLM(),
                  conditions=["suppressed"], suppressed_prompt="world-frame",
                  sampling=Sampling(temperature=0.0, seed=0))
    assert n == 2
    assert {r.think_state for r in read_jsonl(out, GenerationRow)} == {"off", "on"}


def test_unknown_model_raises_instead_of_defaulting_to_na():
    import pytest

    with pytest.raises(KeyError) as e:
        think_states_for("Some-New-Model-9B")
    assert "think_states.yaml" in str(e.value)      


def test_gemma4_toggle_is_explicit_not_a_template_default():


    assert think_states_for("gemma-4-E2B-it") == ["off", "on"]
    assert chat_kwargs("gemma-4-E2B-it", "off") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_every_swept_thinking_model_has_tags_surfaces_can_split():


    import yaml

    from src.analysis.surfaces import TAG_PAIRS
    from src.gen.thinkstate import registry

    known_opens = {o for o, _ in TAG_PAIRS}
    for model, entry in registry().items():
        states = [str(x) for x in entry.get("states", [])]
        if "on" not in states:
            continue
        tag = entry.get("tag") or ""
        assert any(o in tag for o in known_opens), (
            f"{model} is swept think-on with tag {tag!r}, but surfaces.py::TAG_PAIRS knows "
            f"{sorted(known_opens)} — its trace would land in the answer surface")


def test_qwen_ladder_tags_unchanged():
    from src.gen.thinkstate import think_tag

    assert think_tag("Qwen3-8B") == "<think>"
    assert think_tag("Qwen3-1.7B") == "<think>"


def test_the_judge_is_registered_even_though_we_do_not_sweep_it():
    from src.gen.thinkstate import think_tag

    assert think_states_for("Qwen3.6-27B") == ["na"]
    assert think_tag("Qwen3.6-27B") == "<think>"        


def test_yaml_does_not_turn_off_on_into_booleans():


    for m in ("Qwen3-8B", "gemma-4-E2B-it", "Llama-3.1-8B-Instruct"):
        for st in think_states_for(m):
            assert isinstance(st, str), f"{m}: {st!r} is {type(st).__name__}, not str — quote it"
            assert st in ("off", "on", "na")


def _stride(items, i, n):

    return items[i - 1::n]


def test_shards_partition_the_item_set_exactly():
    items = [f"i{k}" for k in range(15130)]
    for n in (2, 3, 4, 8, 16):
        parts = [_stride(items, i, n) for i in range(1, n + 1)]
        union = [x for p in parts for x in p]
        assert sorted(union) == sorted(items), f"n={n} lost or duplicated items"
        assert len(union) == len(set(union)), f"n={n} overlaps"

        assert max(len(p) for p in parts) - min(len(p) for p in parts) <= 1


def test_shard_argument_is_validated():

    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for bad in ("4/2", "0/4", "abc", "1"):
        r = subprocess.run([sys.executable, "scripts/generate.py", "--config", "configs/exp3.yaml",
                            "--model", "Qwen3-8B", "--shard", bad],
                           cwd=root, capture_output=True, text=True)
        assert r.returncode != 0, f"--shard {bad!r} was accepted"
        assert "shard" in (r.stdout + r.stderr).lower(), f"--shard {bad!r} gave no useful message"


def test_coverage_sums_shards_into_one_wave():

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.coverage import _SHARD_SUFFIX

    for name, want in [("sys-scoped-world.shard1of4", "sys-scoped-world"),
                       ("none-baseline.shard12of16", "none-baseline"),
                       ("sys-scoped-world", "sys-scoped-world")]:
        assert _SHARD_SUFFIX.sub("", name) == want


def test_a_shard_skips_cells_an_UNSHARDED_file_already_holds(tmp_path):


    import json
    from src.data.schema import Sampling
    from src.gen import sweep
    from tests.test_runner_resume import MODEL, _DyingLLM, _items

    samp = Sampling(temperature=0.0, top_p=1.0, max_tokens=10, seed=0)
    items = _items(100)
    unsharded = tmp_path / f"{MODEL}__no-meta.jsonl"


    n = sweep.run_sweep(MODEL, "p", items, unsharded, conditions=["suppressed"],
                        suppressed_prompt="no-meta", sampling=samp, llm=_DyingLLM(99),
                        chunk_size=1000)
    assert n == 100


    for i in range(1, 5):
        shard_path = tmp_path / f"{MODEL}__no-meta.shard{i}of4.jsonl"
        written = sweep.run_sweep(MODEL, "p", items[i - 1::4], shard_path,
                                  conditions=["suppressed"], suppressed_prompt="no-meta",
                                  sampling=samp, llm=_DyingLLM(99), chunk_size=1000)
        assert written == 0, f"shard {i} regenerated {written} cell(s) already on disk"

    all_rows = [json.loads(l) for f in sorted(tmp_path.glob("*.jsonl"))
                for l in f.read_text().splitlines() if l.strip()]
    ids = [f"{r['config_hash']}:{r['item_id']}:{r['condition']}:{r['think_state']}" for r in all_rows]
    assert len(ids) == 100 and len(set(ids)) == 100, "the wave must hold each cell exactly once"


def test_shards_still_split_work_when_nothing_exists_yet(tmp_path):

    from src.data.schema import Sampling
    from src.gen import sweep
    from tests.test_runner_resume import MODEL, _DyingLLM, _items

    samp = Sampling(temperature=0.0, top_p=1.0, max_tokens=10, seed=0)
    items = _items(100)
    total = 0
    for i in range(1, 5):
        total += sweep.run_sweep(MODEL, "p", items[i - 1::4],
                                 tmp_path / f"{MODEL}__no-meta.shard{i}of4.jsonl",
                                 conditions=["suppressed"], suppressed_prompt="no-meta",
                                 sampling=samp, llm=_DyingLLM(99), chunk_size=1000)
    assert total == 100, "four shards should generate the whole wave exactly once between them"


def test_context_grows_with_max_tokens_but_never_shrinks_the_old_default():


    from src.data.schema import Sampling
    from src.gen.sweep import context_for

    assert context_for(Sampling(temperature=0.0, seed=0, max_tokens=4096)) == 8192
    assert context_for(Sampling(temperature=0.0, seed=0, max_tokens=8192)) >= 8192 + 2976
    assert context_for(Sampling(temperature=0.0, seed=0, max_tokens=512)) == 8192


def test_max_tokens_is_part_of_config_hash_so_a_new_budget_is_a_new_run():


    from src.data.schema import Item, Sampling, make_item_id
    from src.gen.sweep import run_sweep

    q = "q?"
    item = Item(item_id=make_item_id("c", "forget", q), concept_id="c", concept_class="general",
                concept_subtype="hazard", probe_type="forget", question=q)
    import tempfile
    from pathlib import Path
    hashes = set()
    for mt in (4096, 8192):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "g.jsonl"
            run_sweep("Llama-3.1-8B-Instruct", "/fake", [item], out, conditions=["suppressed"],
                      suppressed_prompt="world-frame", llm=_FakeLLM(),
                      sampling=Sampling(temperature=0.0, seed=0, max_tokens=mt))
            hashes |= {r.config_hash for r in read_jsonl(out, GenerationRow)}
    assert len(hashes) == 2


def test_clamp_context_keeps_a_run_that_fits():
    from src.data.schema import Sampling
    from src.gen.sweep import clamp_context
    import json, tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "config.json").write_text(json.dumps({"max_position_embeddings": 32768}))
        got, msg = clamp_context(12288, d, Sampling(temperature=0.0, top_p=1.0,
                                                   max_tokens=8192, seed=0))
    assert got == 12288 and msg is None


def test_clamp_context_stops_the_failure_that_ate_llama_from_three_runs():


    from src.data.schema import Sampling
    from src.gen.sweep import clamp_context
    import json, tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        pathlib.Path(d, "config.json").write_text(json.dumps({"max_position_embeddings": 8192}))
        got, msg = clamp_context(12288, d, Sampling(temperature=0.0, top_p=1.0,
                                                   max_tokens=8192, seed=0))
    assert got == 8192
    assert msg and "8,192" in msg
    assert "max_tokens exceeds that room" in msg      


def test_clamp_context_does_not_shrink_when_the_config_is_unreadable():

    from src.data.schema import Sampling
    from src.gen.sweep import clamp_context
    got, msg = clamp_context(12288, "/nonexistent/model",
                             Sampling(temperature=0.0, top_p=1.0, max_tokens=8192, seed=0))
    assert got == 12288 and msg is None
