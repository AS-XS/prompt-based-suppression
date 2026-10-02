from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.gen.env import model_default_sampling, resolved_of
from src.provenance import config_hash


JUDGE_CHAT_KWARGS: dict = {}


def default_budget(model_dir: str | Path) -> int | None:


    try:
        cfg = json.loads((Path(model_dir) / "generation_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    v = cfg.get("max_new_tokens")
    return int(v) if isinstance(v, int) and v > 0 else None


def judge_sampling(llm, judge_name: str, max_tokens: int | None):


    from src.gen.thinkstate import needs_raw_special

    sp = model_default_sampling(llm, strict=True)
    sp.max_tokens = max_tokens


    sp.skip_special_tokens = not needs_raw_special(judge_name)
    return sp, resolved_of(sp)


STAGES = {"rating": ("concept", "framing", "mention"), "primitive": ("primitive",)}


def judge_fingerprints(kind: str, judge: str, *, quant=None, tp=None, dtype=None,
                       max_tokens=None, max_model_len=None) -> dict[str, str]:


    from src.judge.prompts import stage_payload

    base = {"kind": kind, "judge": judge, "quant": quant, "tp": tp, "dtype": dtype,
            "decode": "model-default", "chat_template_kwargs": JUDGE_CHAT_KWARGS,
            "max_tokens": max_tokens, "max_model_len": max_model_len}
    return {stage: config_hash({**base, "stage": stage,
                                "prompt": config_hash({"payload": stage_payload(stage)})})
            for stage in STAGES[kind]}


def refuse_mixed_config(prior: list[dict], fingerprint: str, where) -> None:


    other = Counter(d.get("judge_config_hash") or "unrecorded" for d in prior
                    if d.get("judge_config_hash") != fingerprint)
    if other:
        detail = ", ".join(f"{k} x{n}" for k, n in sorted(other.items()))
        raise SystemExit(
            f"{where} already holds judgments from a different judge configuration ({detail}); "
            f"this run is {fingerprint}. Appending would put two instruments in one file, and "
            f"nothing downstream could separate them. Judge into a fresh --out-dir, or move that "
            f"file aside.")
