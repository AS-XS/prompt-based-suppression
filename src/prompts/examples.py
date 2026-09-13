from __future__ import annotations

import random

from src.data.schema import Item
from src.prompts.render import DEFAULT_ANSWER_FORMAT, render_question

_LETTERS = ("A", "B", "C", "D")


LEGACY_SLOTS: tuple[tuple[str, bool], ...] = (("forget", False), ("utility", True))


_SLOT_POLICIES: dict[str, tuple[tuple[str, bool], ...]] = {
    "scoped-directed-fewshot": LEGACY_SLOTS,
    "scoped-directed-fewshot-grouped": (("forget", False), ("neighbor", True), ("utility", True)),


    "scoped-directed-fewshot-grouped-aware":
        (("forget", False), ("neighbor", True), ("utility", True)),
    "scoped-directed-fewshot-grouped-named-broad-aware":
        (("forget", False), ("neighbor", True), ("utility", True)),
}


_CONCEPT_SCOPED = frozenset({"forget", "neighbor"})


def slots_for(prompt_id: str) -> tuple[tuple[str, bool], ...]:


    try:
        return _SLOT_POLICIES[prompt_id]
    except KeyError:
        raise KeyError(f"{prompt_id!r} renders examples but has no slot policy; register one in "
                       f"`_SLOT_POLICIES` (have: {sorted(_SLOT_POLICIES)})") from None


def _wrong_letter(gold: str) -> str:

    i = _LETTERS.index(gold)
    return _LETTERS[(i + 1) % len(_LETTERS)]


def demo_answer(answer_key: str, *, correct: bool) -> str:


    if answer_key not in _LETTERS:
        raise ValueError(f"answer_key {answer_key!r} is not one of {_LETTERS}")
    return f"Answer: {answer_key if correct else _wrong_letter(answer_key)}"


def draw_examples(items: list[Item], concept_id: str, *,
                  slots: tuple[tuple[str, bool], ...] = LEGACY_SLOTS,
                  answer_format: str = DEFAULT_ANSWER_FORMAT,
                  seed: int = 0) -> tuple[list[tuple[str, str]], set[str]]:


    rng = random.Random(f"{seed}:{concept_id}")
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()
    for probe_type, want_correct in slots:
        pool = [it for it in items
                if it.probe_type == probe_type
                and (it.meta or {}).get("format") == "mc"
                and it.answer_key in _LETTERS
                and (probe_type not in _CONCEPT_SCOPED or it.concept_id == concept_id)]
        if not pool:
            return [], set()
        pick = pool[rng.randrange(len(pool))] if len(pool) > 1 else pool[0]
        gold = pick.answer_key
        letter = gold if want_correct else _wrong_letter(gold)
        pairs.append((render_question(pick, answer_format), f"Answer: {letter}"))
        used.add(pick.item_id)
    return pairs, used


def draw_all(items: list[Item], concept_ids, *, prompt_id: str | None = None,
             answer_format: str = DEFAULT_ANSWER_FORMAT,
             seed: int = 0) -> tuple[dict[str, list[tuple[str, str]]], set[str]]:


    slots = slots_for(prompt_id) if prompt_id is not None else LEGACY_SLOTS
    by_concept: dict[str, list[tuple[str, str]]] = {}
    used: set[str] = set()
    for cid in sorted(set(concept_ids)):
        pairs, ids = draw_examples(items, cid, slots=slots, answer_format=answer_format, seed=seed)
        if not pairs:
            continue
        by_concept[cid] = pairs
        used |= ids
    return by_concept, used
