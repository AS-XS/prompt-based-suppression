from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

ON_TARGET = "forget"


_REFUSAL = re.compile(
    r"\b(i (can(no|')t|cannot|am unable|don'?t know|do not know)"
    r"|unable to (answer|determine|say)"
    r"|not enough information|insufficient information"
    r"|underspecified|cannot be determined|impossible to (say|determine|answer)"
    r"|without (that|this) (knowledge|information)"
    r"|no information about)\b",
    re.IGNORECASE,
)


def looks_like_refusal(text: str | None) -> bool:

    return bool(_REFUSAL.search(text or ""))


def classify(probe_type: str, base_correct: bool | None, supp_correct: bool | None,
             supp_parsed: bool) -> str:

    if base_correct is None or not base_correct:
        return "already_wrong"          
    if not supp_parsed:
        return "unparsed"               
    if probe_type == ON_TARGET:
        return "leaked_correct" if supp_correct else "held"
    return "intact" if supp_correct else "collateral"


UNDESIRED = {"leaked_correct", "collateral", "unparsed"}


def pair_outcomes(scored: Iterable[Mapping], baseline_prompt: str = "none-baseline") -> list[dict]:


    base: dict[tuple, Mapping] = {}
    supp: list[Mapping] = []
    for r in scored:
        key = (r.get("model"), r.get("think_state"), r.get("item_id"))
        if r.get("prompt_id") == baseline_prompt:
            base[key] = r
        else:
            supp.append(r)

    out = []
    for r in supp:
        b = base.get((r.get("model"), r.get("think_state"), r.get("item_id")))
        if b is None:
            continue
        outcome = classify(r.get("probe_type"), b.get("correct"), r.get("correct"),
                           bool(r.get("parsed")))
        out.append({**r, "outcome": outcome, "baseline_correct": b.get("correct"),
                    "baseline_pred": b.get("pred")})
    return out


def outcome_table(paired: Iterable[Mapping],
                  by: Sequence[str] = ("model", "prompt_id", "probe_type")):


    groups: dict[tuple, Counter] = defaultdict(Counter)
    for r in paired:
        groups[tuple(r.get(k) for k in by)][r["outcome"]] += 1
    rows = []
    for key, c in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n = sum(c.values())
        eligible = n - c["already_wrong"]
        bad = sum(c[k] for k in UNDESIRED)
        row = {**dict(zip(by, key)), "n": n, "n_eligible": eligible,
               "undesired_rate": round(bad / eligible, 4) if eligible else None}
        for k in ("held", "leaked_correct", "intact", "collateral", "unparsed", "already_wrong"):
            row[k] = c[k]
        rows.append(row)
    return rows


def stratified_sample(paired: Iterable[Mapping], per_cell: int = 5,
                      by: Sequence[str] = ("prompt_id", "probe_type", "outcome"),
                      seed: int = 0, only: Sequence[str] | None = None) -> list[dict]:


    keep = set(only) if only is not None else UNDESIRED
    cells: dict[tuple, list] = defaultdict(list)
    for r in paired:
        if r["outcome"] in keep:
            cells[tuple(r.get(k) for k in by)].append(r)
    rng = random.Random(seed)
    out = []
    for key in sorted(cells, key=lambda t: [str(x) for x in t]):
        rows = cells[key]
        rng.shuffle(rows)
        out.extend(rows[:per_cell])
    return out
