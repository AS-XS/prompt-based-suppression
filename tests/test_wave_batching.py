from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.prompts.labels import LADDER_PROMPT_IDS  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")


def test_several_waves_share_one_engine_and_still_write_one_file_each(tmp_path):


    from src.data.schema import GenerationRow, Item, Sampling, make_item_id
    from src.gen.runner import read_jsonl
    from src.gen.sweep import run_sweep

    class _Out:
        def __init__(self):
            self.outputs = [type("O", (), {"text": "ANSWER: B", "token_ids": [1],
                                           "finish_reason": "stop"})()]
            self.prompt_token_ids = [0]

    class _Recorder:
        def __init__(self):
            self.seen = []

        def chat(self, messages, sp, **kw):
            self.seen.append(sp)
            return [_Out() for _ in messages]

    q = "What is a cell?"
    item = Item(item_id=make_item_id("wmdp-bio", "forget", q), concept_id="wmdp-bio",
                concept_class="general", concept_subtype="hazard", probe_type="forget", question=q)
    llm, marker = _Recorder(), object()
    hashes = set()
    for wave in ("scoped-world@generic", "scoped-world@keywords"):
        out = tmp_path / f"M__{wave}.jsonl"
        n = run_sweep("Llama-3.1-8B-Instruct", "/fake", [item], out, conditions=["suppressed"],
                      suppressed_prompt=wave, sampling=Sampling(temperature=0.0, seed=0),
                      llm=llm, sampling_params=marker)
        assert n == 1
        rows = list(read_jsonl(out, GenerationRow))
        assert [r.prompt_id for r in rows] == [wave]
        hashes.add(rows[0].config_hash)
    assert len(hashes) == 2, "two prompts must not share a config_hash"
    assert llm.seen == [marker, marker], "an injected engine must get the injected SamplingParams"


def test_multiple_waves_with_multiple_conditions_is_refused(tmp_path):


    r = subprocess.run([sys.executable, "scripts/generate.py", "--config", "configs/wmdp1.yaml",
                        "--model", "M", "--prompt-suppressed", "a@x", "b@y",
                        "--conditions", "baseline", "suppressed"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0
    assert "single condition" in (r.stdout + r.stderr)


def test_coverage_does_not_invent_gaps_when_models_got_different_waves(tmp_path):


    import json

    gens = tmp_path / "results" / "r" / "generations"
    gens.mkdir(parents=True)
    row = {"model": "M", "prompt_id": "", "think_state": "na", "item_id": "i0", "config_hash": "h",
           "condition": "suppressed", "probe_type": "forget", "finish_reason": "stop",
           "raw_output": "Answer: A"}

    for model, waves in (("zephyr-7b-beta", ["none-baseline", "a@generic", "a@keywords"]),
                         ("Meta-Llama-3-8B-Instruct", ["none-baseline", "a@generic"])):
        for w in waves:
            (gens / f"{model}__{w}.jsonl").write_text(
                json.dumps({**row, "model": model, "prompt_id": w}) + "\n", encoding="utf-8")

    def run(*extra):

        return subprocess.run([sys.executable, str(ROOT / "scripts" / "coverage.py"),
                               "--run", "r", *extra],
                              cwd=tmp_path, capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(ROOT)})

    r = run()
    assert "not submitted" in r.stdout, "an absent wave must not read as a gap by default"
    assert "--prompt-suppressed a@keywords" not in r.stdout, \
        "must NOT offer to generate a wave nobody asked for"
    assert "not submitted" in r.stdout and "MISSING" not in r.stdout

    r2 = run("--models", "Meta-Llama-3-8B-Instruct", "--waves", "none-baseline", "a@keywords")
    assert "MISSING" in r2.stdout, "--waves states the intent, so absence becomes a real gap"
    assert "--prompt-suppressed a@keywords" in r2.stdout


def test_a_wave_estimate_scales_with_states_and_token_budget():


    from src.gen.cost import minutes_per_wave

    quiet = minutes_per_wave("Meta-Llama-3-8B-Instruct")
    both = minutes_per_wave("Qwen3-8B")
    assert quiet < 20 < both, "a think-on arm dominates the wave"
    assert minutes_per_wave("Olmo-3-7B-Think") > both, "always-on has no think-off half"

    at_8k = minutes_per_wave("Qwen3-8B", max_tokens=8192)
    assert both < at_8k < 2 * both


def test_size_tags_match_whole_tokens_not_substrings():


    from src.gen.cost import _size_factor, minutes_per_wave

    assert _size_factor("Qwen3-14B") == 2.0
    assert _size_factor("OLMo-2-1124-13B-Instruct") == 1.7
    assert _size_factor("Mistral-Small-24B-Instruct-2501") == 3.0
    assert _size_factor("gemma-4-E4B-it") == 0.7, "its own tag now, not the `4B` inside `E4B`"
    ladder = [minutes_per_wave(m) for m in ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B",
                                            "Qwen3-32B")]
    assert all(a < b for a, b in zip(ladder, ladder[1:])), ladder


def test_waves_per_job_never_returns_zero():


    from src.gen.cost import waves_per_job

    assert waves_per_job("Olmo-3-7B-Think", budget_hours=1.0) == 1
    assert waves_per_job("Meta-Llama-3-8B-Instruct", budget_hours=9.0) > 1


def test_no_roster_model_is_silently_costed_as_an_8B():


    from src.gen.cost import _size_factor

    assert _size_factor("gemma-4-31B-it") == 3.9
    assert _size_factor("Meta-Llama-3-70B-Instruct") == 8.75
    assert _size_factor("Olmo-3.1-32B-Think") == 4.0

    roster = [ln.split("#")[0].strip().split("/")[-1]
              for ln in (ROOT / "scripts" / "models.txt").read_text().splitlines()
              if ln.strip() and not ln.strip().startswith("#")]

    import re
    missing = [m for m in roster
               if (g := re.search(r"-(\d+(?:\.\d+)?)[Bb]\b", m))
               and float(g.group(1)) >= 9 and _size_factor(m) == 1.0]
    assert not missing, f"these size themselves as an 8B: {missing}"


