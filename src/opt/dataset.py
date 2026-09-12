from __future__ import annotations

from pathlib import Path

from src.io import read_dicts as _read_jsonl
from src.analysis.accuracy import extract_letter_tiered
from src.analysis.surfaces import answer_of
from src.data.split import read_val_ids
from src.prompts import render_question
from src.prompts.examples import demo_answer


def build_examples(items_path: Path, baseline_path: Path, split_path: Path, *,
                   think_state: str, concepts: tuple[str, ...] | None = None,
                   probe_types: tuple[str, ...] = ("forget",), limit: int | None = None):


    import dspy

    val_ids = read_val_ids(split_path)
    keys: dict[str, dict] = {}
    for it in _read_jsonl(items_path):
        meta = it.get("meta") or {}
        if meta.get("format") != "mc" or not it.get("answer_key"):
            continue
        if it.get("probe_type") not in probe_types:
            continue
        if concepts and it.get("concept_id") not in concepts:
            continue
        if it.get("probe_type") == "forget" and it["item_id"] not in val_ids:
            continue
        keys[it["item_id"]] = it

    base: dict[str, bool] = {}
    for g in _read_jsonl(baseline_path):
        if g.get("think_state") != think_state or g.get("condition") != "baseline":
            continue
        it = keys.get(g["item_id"])
        if not it:
            continue
        pred, _ = extract_letter_tiered(
            answer_of(g.get("raw_output"), expect_trace=g.get("think_state") == "on"))
        base[g["item_id"]] = pred is not None and pred == it["answer_key"]

    examples = []
    for item_id, it in keys.items():
        if not base.get(item_id):
            continue
        from src.data.schema import Item


        examples.append(
            dspy.Example(question=render_question(Item(**it)), item_id=item_id,
                         answer_key=it["answer_key"], probe_type=it["probe_type"],
                         response=demo_answer(it["answer_key"],
                                              correct=it["probe_type"] != "forget"),
                         concept_id=it.get("concept_id"), baseline_correct=True)
            .with_inputs("question"))
    examples.sort(key=lambda e: e.item_id)
    return _stratified_head(examples, limit) if limit else examples


def _stratified_head(examples, limit: int):


    by_arm: dict[str, list] = {}
    for e in examples:
        by_arm.setdefault(e.probe_type, []).append(e)
    if len(by_arm) <= 1:
        return examples[:limit]
    total = len(examples)
    out: list = []
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        take = max(1, round(limit * len(rows) / total))
        out += rows[:take]

    while len(out) > limit and len(out) > len(by_arm):
        biggest = max(sorted(by_arm), key=lambda a: sum(1 for e in out if e.probe_type == a))
        for i in range(len(out) - 1, -1, -1):
            if out[i].probe_type == biggest:
                del out[i]
                break
    out.sort(key=lambda e: e.item_id)
    return out


def stratified_split(examples, train_frac: float = 0.6):


    by_arm: dict[str, list] = {}
    for e in examples:
        by_arm.setdefault(e.probe_type, []).append(e)
    train, dev = [], []
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        cut = max(1, int(len(rows) * train_frac)) if len(rows) > 1 else 1
        train += rows[:cut]
        dev += rows[cut:]
    return train, (dev or examples[:1])
