from __future__ import annotations

from src.judge.prompts import primitive_messages
from src.judge.run import parse_primitives, run_primitive_judge

_VALID = ('{"rationale":"names DNA directly","semantic_leak":true,"leak_type":"direct",'
          '"meta_reference":false,"fluent":true,"guessing":false}')


class _Out:
    def __init__(self, t): self.text = t


class _Gen:
    def __init__(self, t): self.outputs = [_Out(t)]


class _FakeLLM:
    def __init__(self, payload): self.payload = payload; self.seen = None
    def chat(self, conv, sp): self.seen = conv; return [_Gen(self.payload)]


def test_parse_primitives_valid():
    d = parse_primitives(_VALID)
    assert d["semantic_leak"] is True and d["leak_type"] == "direct" and d["fluent"] is True
    assert d["guessing"] is False               


def test_parse_primitives_rejects_missing_guessing():


    no_guess = ('{"rationale":"x","semantic_leak":false,"leak_type":null,'
                '"meta_reference":false,"fluent":true}')
    assert parse_primitives(no_guess) is None


def test_parse_primitives_rejects_garbage_and_partial():
    assert parse_primitives("not json at all") is None
    assert parse_primitives('{"semantic_leak": true}') is None  


def test_run_primitive_judge_stores_raw_and_primitives():
    llm = _FakeLLM(_VALID)
    res = run_primitive_judge(llm, None, concept="biology", sentence="cells store DNA", surface="answer")
    assert res["surface"] == "answer" and res["parsed_ok"] is True
    assert res["semantic_leak"] is True and res["meta_reference"] is False and res["guessing"] is False
    assert res["raw_judge_json"] == _VALID  


def test_run_primitive_judge_bad_json_is_none_not_crash():
    res = run_primitive_judge(_FakeLLM("garbage"), None, concept="biology", sentence="x", surface="trace")
    assert res["parsed_ok"] is False and res["semantic_leak"] is None and res["surface"] == "trace"


def test_primitive_messages_surface_wording():
    assert "reasoning trace" in primitive_messages("biology", "x", "trace")[0]["content"]
    assert "answer" in primitive_messages("biology", "x", "answer")[0]["content"]


def test_shard_stride_partitions_rows_exactly():


    rows = list(range(1000))
    n = 4
    shards = [rows[i - 1::n] for i in range(1, n + 1)]
    assert sorted(x for s in shards for x in s) == rows
    assert all(len(s) == 250 for s in shards)
    assert shards[0][:3] == [0, 4, 8]

    assert sum(len(list(range(10))[i - 1::3]) for i in range(1, 4)) == 10


def test_free_text_arm_is_not_described_to_the_judge_as_multiple_choice():


    mc = primitive_messages("biology", "because cells divide", "explanation",
                            concept_class="general")[0]["content"]
    free = primitive_messages("Stephen King", "John Grisham.", "explanation",
                              concept_class="isolated")[0]["content"]

    assert "multiple-choice" in mc and "its choice" in mc
    assert "multiple-choice" not in free, "the RWKU arm is still being told it answered an MC question"
    assert "its answer" in free

    assert "brief explanation of why it chose" in mc
    assert "brief explanation of why it chose" not in free


def test_trace_surface_keeps_its_framing_on_both_arms():
    for cls in ("general", "isolated"):
        t = primitive_messages("biology", "hmm", "trace", concept_class=cls)[0]["content"]
        assert "internal reasoning trace" in t and "not its final answer" in t


def test_run_primitive_judge_passes_the_concept_class_through():
    llm = _FakeLLM(_VALID)
    run_primitive_judge(llm, None, concept="Stephen King", sentence="John Grisham.",
                        surface="explanation", concept_class="isolated")
    assert "multiple-choice" not in llm.seen[0][0]["content"]
