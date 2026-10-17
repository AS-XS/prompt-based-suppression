from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, NamedTuple, Sequence

BASELINE = "none-baseline"
UNLEARN_METHODS = {"rmu-": "RMU", "elm-": "ELM", "r2mu-": "R2MU"}
_CONCEPT_TOKENS = {"bio": "wmdp-bio", "cyber": "wmdp-cyber"}


class Cell(NamedTuple):
    run: str
    model: str
    concept_id: str
    prompt_id: str
    think_state: str


def method_of(model: str) -> str | None:

    return next((m for p, m in UNLEARN_METHODS.items() if model.startswith(p)), None)


def target_concept(model: str) -> str:


    tokens = set(model.split("-"))
    hits = sorted(c for t, c in _CONCEPT_TOKENS.items() if t in tokens)
    if len(hits) != 1:
        raise ValueError(f"cannot tell which concept {model!r} removes (found {hits or 'none'})")
    return hits[0]


def arm_of(cell: Cell) -> str:
    if method_of(cell.model):
        return "unlearned"
    return "baseline" if cell.prompt_id == BASELINE else "suppressed"


def rank_key(seed: int, concept_id: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}:{concept_id}:{item_id}".encode()).hexdigest()


def draw(cells: Mapping[Cell, Iterable[str]], per_concept: int,
         seed: int) -> tuple[dict[Cell, list[str]], list[tuple[Cell, int, int]]]:


    picks, short = {}, []
    for cell, items in cells.items():
        ordered = sorted(items, key=lambda i: rank_key(seed, cell.concept_id, i))
        picks[cell] = ordered[:per_concept]
        if len(ordered) < per_concept:
            short.append((cell, per_concept, len(ordered)))
    return picks, short


def file_part(cell: Cell, prompt_run: str) -> str:


    return f"{cell.run}__{cell.model}" if cell.run == prompt_run else "unlearned"


def file_order(picks: Mapping[Cell, Sequence[str]], prompt_run: str) -> dict[str, list[tuple]]:


    out: dict[str, list[tuple]] = {}
    for cell, items in picks.items():
        for rank, item in enumerate(items):
            out.setdefault(file_part(cell, prompt_run), []).append((rank, cell, item))
    for rows in out.values():
        rows.sort(key=lambda t: (t[0], tuple(t[1]), t[2]))
    return out


def appended_tail(existing: Sequence[str], drawn: Sequence[str]) -> list[str]:


    if list(drawn[:len(existing)]) != list(existing):
        at = next((i for i, (a, b) in enumerate(zip(existing, drawn)) if a != b),
                  min(len(existing), len(drawn)))
        raise ValueError(f"the new draw does not extend this file: they first differ at row "
                         f"{at + 1} of {len(existing)}. The source generations changed since it was "
                         f"drawn (e.g. a model's waves were still generating), or the seed differs.")
    return list(drawn[len(existing):])


def same_instrument(run_base: Mapping[str, set], prompt_base: Mapping[str, set]) -> tuple[bool, str]:


    if not run_base:
        return False, "no base-model rows to compare"
    for model, hashes in run_base.items():
        if model not in prompt_base:
            return False, f"its base model {model} is not in the prompt run"
        if hashes != prompt_base[model]:
            return False, (f"its {model} baseline is configuration {sorted(hashes)}, the prompt "
                           f"run's is {sorted(prompt_base[model])}")
    return True, "same instrument"
