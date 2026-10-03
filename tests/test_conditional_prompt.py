import pytest
import yaml

from src.data.schema import GenerationRow, Item, Sampling, make_item_id
from src.gen.runner import read_jsonl
from src.gen.sweep import run_sweep
from src.prompts import ALL_PROMPT_IDS, build_messages, prompt_fingerprint
from src.prompts import conditional
from src.prompts.labels import LABEL_SCHEMES


@pytest.mark.parametrize("scheme", ["specific", "keywords"])
@pytest.mark.parametrize("concept", ["wmdp-bio", "wmdp-cyber"])
def test_registered_candidate_renders_the_requested_scope(scheme, concept):
    assert "scoped-conditional" in ALL_PROMPT_IDS
    question = "Question with {literal braces}\nAnswer: X / Why: explanation"
    messages = build_messages(f"scoped-conditional@{scheme}", concept, question)
    assert len(messages) == 1 and messages[0]["role"] == "user"
    text = messages[0]["content"]
    assert text.endswith(question)
    assert LABEL_SCHEMES[scheme][concept] in text
    assert concept not in text and "{concept}" not in text


def test_candidate_edits_change_only_candidate_fingerprints(monkeypatch):
    old_ids = [p for p in ALL_PROMPT_IDS if p != "scoped-conditional"]
    old_ids += ["scoped-directed-aware@keywords", "scoped-assess-aware@specific"]
    before = {p: prompt_fingerprint(p) for p in old_ids}
    candidates = ["scoped-conditional@specific", "scoped-conditional@keywords"]
    hashes = {p: prompt_fingerprint(p) for p in candidates}
    assert len(set(hashes.values())) == 2
    monkeypatch.setattr(conditional, "_INSTRUCTION", conditional._INSTRUCTION + " Changed.")
    assert all(prompt_fingerprint(p) != hashes[p] for p in candidates)
    assert {p: prompt_fingerprint(p) for p in old_ids} == before


def test_candidate_rejects_unregistered_demonstrations():
    with pytest.raises(ValueError, match="no demonstrations"):
        build_messages("scoped-conditional@specific", "wmdp-bio", "Q?", [("Q", "Answer: A")])


def test_candidate_waves_resume_independently_and_preserve_raw_output(tmp_path):
    class FakeLLM:
        calls = 0

        def chat(self, messages, sampling, **kwargs):
            self.calls += 1
            answer = type("Answer", (), {"text": "Answer: B\nWhy: uncertain.",
                                         "token_ids": [1, 2], "finish_reason": "stop"})()
            output = type("Output", (), {"outputs": [answer], "prompt_token_ids": [0]})()
            return [output for _ in messages]

    item = Item(item_id=make_item_id("wmdp-bio", "forget", "Q?"),
                concept_id="wmdp-bio", concept_class="general", concept_subtype="hazard",
                probe_type="forget", question="Q?")
    llm = FakeLLM()
    hashes = set()
    for scheme in ("specific", "keywords"):
        prompt = f"scoped-conditional@{scheme}"
        path = tmp_path / f"M__{prompt}.jsonl"
        kwargs = dict(conditions=["suppressed"], suppressed_prompt=prompt,
                      sampling=Sampling(max_tokens=16384), llm=llm, sampling_params=object())
        assert run_sweep("Meta-Llama-3-8B-Instruct", "/fake", [item], path, **kwargs) == 1
        raw = path.read_bytes()
        assert run_sweep("Meta-Llama-3-8B-Instruct", "/fake", [item], path, **kwargs) == 0
        assert path.read_bytes() == raw
        row, = read_jsonl(path, GenerationRow)
        assert row.prompt_id == prompt and row.condition == "suppressed"
        assert row.raw_output == "Answer: B\nWhy: uncertain."
        hashes.add(row.config_hash)
    assert len(hashes) == 2 and llm.calls == 2


def test_wmdp8_extension_keeps_existing_instrument():
    from pathlib import Path

    cfg = yaml.safe_load(Path("configs/wmdp8.yaml").read_text())
    assert cfg["run"] == "wmdp8"
    assert cfg["sampling"] == {"max_tokens": 16384}
    assert cfg["waves"][-2:] == ["scoped-conditional@specific", "scoped-conditional@keywords"]
    assert len(cfg["waves"]) == len(set(cfg["waves"])) == 26
