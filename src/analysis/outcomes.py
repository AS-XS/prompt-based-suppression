from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.data.identity import generation_id_of
from src.analysis.accuracy import score_generations
from src.analysis.judge_sample import outcome_of
from src.io import read_dicts
from src.paths import generations_dir

BASELINE = "none-baseline"


def baseline_rows(prompt_run: str, wanted: set[tuple[str, str, str]]) -> dict[tuple, dict]:


    out: dict[tuple, dict] = {}
    for model in sorted({m for m, _, _ in wanted}):
        for f in sorted(generations_dir(prompt_run).glob(f"{model}__{BASELINE}*.jsonl")):
            for g in read_dicts(f):
                key = (g.get("model"), g.get("item_id"), g.get("think_state"))
                if key in wanted and key not in out:
                    out[key] = g
    return out


def outcomes(sample: Mapping[str, Mapping], manifest_rows: Mapping[str, Mapping],
             item_index: Mapping[str, Mapping], prompt_run: str) -> dict[str, dict]:


    def base_of(gid: str) -> str:
        meta = manifest_rows.get(gid, {})
        return meta.get("base_model") or sample[gid]["model"]

    wanted = {(base_of(g), r["item_id"], r["think_state"]) for g, r in sample.items()}
    base = baseline_rows(prompt_run, wanted)
    own = {(r["model"], r["item_id"], r["prompt_id"], r["think_state"]): r
           for r in score_generations(sample.values(), item_index)}
    ref = {(r["model"], r["item_id"], r["think_state"]): r
           for r in score_generations(base.values(), item_index)}
    out = {}
    for gid, row in sample.items():
        mine = own.get((row["model"], row["item_id"], row["prompt_id"], row["think_state"]), {})
        b = ref.get((base_of(gid), row["item_id"], row["think_state"]), {})
        is_baseline = (manifest_rows.get(gid, {}).get("arm")
                       or row.get("condition")) == "baseline"
        braw = base.get((base_of(gid), row["item_id"], row["think_state"]))
        out[gid] = {"pred": mine.get("pred"), "parsed": mine.get("parsed"),
                    "correct": mine.get("correct"), "parse_tier": mine.get("parse_tier"),
                    "baseline_generation_id": generation_id_of(braw) if braw else None,
                    "baseline_pred": b.get("pred"), "baseline_correct": b.get("correct"),
                    "outcome": None if is_baseline else outcome_of(b.get("correct"),
                                                                   mine.get("correct"))}
    return out


def item_details(items_file: Path) -> dict[str, dict]:

    out = {}
    for it in read_dicts(items_file):
        choices = (it.get("meta") or {}).get("choices") or []
        key = it.get("answer_key")
        idx = "ABCDEFGHIJ".find(key) if key else -1
        out[it["item_id"]] = {"question": it.get("question"), "choices": choices,
                              "answer_key": key,
                              "answer_text": choices[idx] if 0 <= idx < len(choices) else None}
    return out
