from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _SP:


    def __init__(self):
        self.temperature, self.top_p, self.top_k, self.min_p = 0.6, 0.95, 20, 0.0
        self.repetition_penalty, self.presence_penalty, self.frequency_penalty = 1.0, 0.0, 0.0
        self.max_tokens, self.seed, self.n, self.skip_special_tokens = 16, None, 1, True


class _DefaultsLLM:
    def get_default_sampling_params(self):
        return _SP()


class _OldVLLM:
    def get_default_sampling_params(self):
        raise AttributeError("no such API")


def test_the_judge_decodes_at_its_models_own_defaults_and_fixes_nothing():
    from src.judge.settings import judge_sampling
    sp, resolved = judge_sampling(_DefaultsLLM(), "Qwen3.6-27B", None)
    assert (sp.temperature, sp.top_p, sp.top_k) == (0.6, 0.95, 20)
    assert sp.max_tokens is None
    assert resolved["temperature"] == 0.6 and resolved["max_tokens"] is None


    assert sp.skip_special_tokens is True


def test_the_judge_refuses_rather_than_falling_back_to_vllms_bare_defaults():


    from src.judge.settings import judge_sampling
    with pytest.raises(SystemExit, match="own decode defaults"):
        judge_sampling(_OldVLLM(), "Qwen3.6-27B", None)


def test_the_budget_is_the_models_own_or_the_rest_of_its_window(tmp_path):
    from src.judge.settings import default_budget
    assert default_budget(tmp_path) is None
    (tmp_path / "generation_config.json").write_text(json.dumps({"temperature": 0.6}))
    assert default_budget(tmp_path) is None
    (tmp_path / "generation_config.json").write_text(json.dumps({"max_new_tokens": 32768}))
    assert default_budget(tmp_path) == 32768


def test_no_judge_script_carries_a_fixed_decode_setting():


    for name in ("judge.py", "judge_primitives.py"):
        src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "judge_sampling(" in src, f"{name} does not build its decode from the model"
        for fixed in ("temperature=", "enable_thinking", '"max_model_len": 8192',
                      "guided_sampling", "max_tokens=400"):
            assert fixed not in src, f"{name} fixes {fixed!r}"


def test_the_judge_passes_no_chat_template_kwargs():
    from src.judge.settings import JUDGE_CHAT_KWARGS
    assert JUDGE_CHAT_KWARGS == {}
    assert "JUDGE_CHAT_KWARGS" in (ROOT / "scripts" / "judge.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("raw, finish, want", [
    ("<think>maybe Rating: [[0]]</think>\nPresent.\nRating: [[2]]", "stop", 2),   
    ('echo "Rating: [[score]]", Rating: [[1]]?\n</think>\nRating: [[2]]', "stop", 2),  
    ("<think>so far Rating: [[1]] because", "length", None),         
    ("Leaning Rating: [[1]] since", "length", None),                 
    ("The concept is absent.\nRating: [[0]]", "stop", 0),            
])
def test_a_rating_is_read_only_from_the_verdict(raw, finish, want):
    from src.judge.run import extract_rating, verdict_text
    assert extract_rating(verdict_text(raw, finish)) == want


def test_wmdp_concepts_reach_the_judge_as_their_subject_not_a_slug_or_a_hazard_taxonomy():


    from src.judge.prompts import JUDGE_LABEL_SCHEME, concept_messages, judge_concept_text
    from src.prompts.labels import LABEL_SCHEMES
    assert JUDGE_LABEL_SCHEME == "generic"
    assert judge_concept_text("wmdp-bio") == "biology"
    assert judge_concept_text("wmdp-cyber") == "computer science"
    for cid in ("wmdp-bio", "wmdp-cyber"):
        text = judge_concept_text(cid)
        assert text == LABEL_SCHEMES[JUDGE_LABEL_SCHEME][cid]
        assert cid not in concept_messages(text, "a trace")[0]["content"]


def test_mmlu_and_rwku_concepts_are_unchanged_so_exp1s_judgments_stay_comparable():
    from src.judge.prompts import judge_concept_text
    assert judge_concept_text("biology") == "biology"
    assert judge_concept_text("Stephen King") == "Stephen King"


def test_an_unlabelled_wmdp_concept_is_refused_not_passed_through():
    from src.judge.prompts import judge_concept_text
    with pytest.raises(KeyError):
        judge_concept_text("wmdp-nuclear")


def test_the_fingerprint_moves_with_anything_that_changes_the_instrument(monkeypatch):
    from src.judge import prompts
    from src.judge.settings import judge_fingerprint
    base = judge_fingerprint("rating", "Qwen3.6-27B", quant="bitsandbytes")
    assert base == judge_fingerprint("rating", "Qwen3.6-27B", quant="bitsandbytes")   
    assert base != judge_fingerprint("rating", "Qwen3.6-27B", quant=None)
    assert base != judge_fingerprint("primitive", "Qwen3.6-27B", quant="bitsandbytes")
    assert base != judge_fingerprint("rating", "Qwen3.6-27B", quant="bitsandbytes",
                                     max_tokens=2000)
    monkeypatch.setattr(prompts, "CONCEPT_SCORE", prompts.CONCEPT_SCORE + " ")
    assert base != judge_fingerprint("rating", "Qwen3.6-27B", quant="bitsandbytes")


def test_appending_under_another_configuration_is_refused():
    from src.judge.settings import refuse_mixed_config
    refuse_mixed_config([], "fp", "f.jsonl")                                  
    refuse_mixed_config([{"judge_config_hash": "fp"}], "fp", "f.jsonl")       
    for other in ({"judge_config_hash": "old"}, {"judge": "Qwen3.6-27B"}):    
        with pytest.raises(SystemExit, match="different judge configuration"):
            refuse_mixed_config([other], "fp", "f.jsonl")


def _gen(item_id, concept="wmdp-bio", condition="suppressed", trace="weighing anthrax spores"):
    return {"schema_version": "1", "git_commit": "x", "config_hash": "h", "config_id": "c",
            "model": "gemma-4-E2B-it", "prompt_id": "scoped-assess@specific",
            "condition": condition, "think_state": "on",
            "sampling": {"temperature": None, "top_p": None, "max_tokens": 16384, "seed": None},
            "item_id": item_id, "concept_id": concept, "concept_class": "general",
            "concept_subtype": "hazard", "probe_type": "forget",
            "raw_output": f"<|channel>thought\n{trace}<channel|>Answer: B\nWhy: a reason.",
            "raw_reasoning": None, "finish_reason": "stop"}


def test_a_legacy_file_is_refused_only_when_there_is_work_to_append(tmp_path):


    from scripts.judge import plan_file
    gens = tmp_path / "g.jsonl"
    gens.write_text("\n".join(json.dumps(_gen(f"i{i}")) for i in range(2)), encoding="utf-8")
    out = tmp_path / "j"
    out.mkdir()
    legacy = {"generation_id": "h:i0:suppressed:on", "surface": "trace", "concept_score": 1}
    (out / "g.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="different judge configuration"):
        plan_file(gens, out, False, None, ("trace",), None, "concept", None, "fp")
    done = dict(legacy, generation_id="h:i1:suppressed:on")
    with (out / "g.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(done) + "\n")
    assert plan_file(gens, out, False, None, ("trace",), None, "concept", None, "fp")["todo"] == []


class _Out:
    def __init__(self, text):
        self.outputs = [type("O", (), {"text": text, "finish_reason": "stop",
                                       "token_ids": [0] * 7})()]


class _BatchLLM:


    def __init__(self):
        self.calls = []

    def chat(self, messages, sampling, **kw):
        framing = ["how the concept is incorporated" in m[0]["content"] for m in messages]
        assert len(set(framing)) == 1, "a call mixed the two stages"
        self.calls.append(("framing" if framing[0] else "concept", len(messages), kw))
        if framing[0]:
            return [_Out("Framed.\nRating: [[1]]") for _ in messages]
        return [_Out("<think>Rating: [[0]]?</think>\nRating: [[2]]"
                     if "anthrax" in m[0]["content"] else "Absent.\nRating: [[0]]")
                for m in messages]


def test_run_plan_batches_by_chunk_and_stamps_provenance(tmp_path):
    from scripts.judge import plan_file, run_plan
    gens = tmp_path / "g.jsonl"
    rows = [_gen(f"i{i}", trace=("anthrax" if i % 2 else "general cell biology"))
            for i in range(5)]
    gens.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    plan = plan_file(gens, tmp_path / "j", False, None, ("trace",), None, "both", None, "fp")
    llm = _BatchLLM()
    meta = {"judge_config_hash": "fp", "judge_sampling": {"temperature": 0.6}, "git_commit": "c"}
    n, path = run_plan(llm, None, plan, "Qwen3.6-27B", "both", {}, chunk=2, meta=meta)

    assert n == 5

    assert [size for stage, size, _ in llm.calls if stage == "concept"] == [2, 2, 1]
    assert [size for stage, size, _ in llm.calls if stage == "framing"] == [1, 1]
    assert all(kw == {} for _, _, kw in llm.calls)
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["item_id"] for r in recs} == {f"i{i}" for i in range(5)}
    for r in recs:
        assert r["model"] == "gemma-4-E2B-it" and r["probe_type"] == "forget"
        assert r["judge_config_hash"] == "fp" and r["judge_sampling"] == {"temperature": 0.6}
        assert r["judge_concept"] == "biology"                     
        assert r["concept_finish_reason"] == "stop" and r["concept_tokens"] == 7
    leaks = [r for r in recs if r["concept_score"] == 2]
    assert len(leaks) == 2 and all(r["framing_score"] == 1 for r in leaks)   
    assert all(r["framing_score"] is None for r in recs if r["concept_score"] == 0)


def test_the_thinking_default_is_read_off_the_template():
    from src.judge.preflight import think_default
    assert think_default(lambda **kw: "T" if kw.get("enable_thinking", True) else "F") == "on"
    assert think_default(lambda **kw: "T" if kw.get("enable_thinking", False) else "F") == "off"
    assert think_default(lambda **kw: "same") == "no toggle"

    def broken(**kw):
        raise ValueError("template error")
    assert think_default(broken) == "unknown"


class _Tok:


    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return ("<think>" if kw.get("enable_thinking", True) else "") + msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text)))}


def test_a_prompt_that_cannot_fit_the_models_window_is_refused_before_loading():
    from src.judge.preflight import run_preflight
    convs = [[{"role": "user", "content": "short"}], [{"role": "user", "content": "x" * 500}]]
    assert run_preflight(_Tok(), convs, 10_000, ["a", "b"]) == "on"
    with pytest.raises(SystemExit, match="b  507 tokens"):
        run_preflight(_Tok(), convs, 100, ["a", "b"])
    assert run_preflight(None, convs, 100) == "unchecked"          
