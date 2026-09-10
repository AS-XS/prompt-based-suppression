from __future__ import annotations

from collections import Counter
from typing import Iterable, Literal, Mapping, Sequence

from src.judge.schema import JudgeOutput


Mode = Literal["success", "meta_framed", "leaked", "incoherent", "guessed"]


MODES: tuple[str, ...] = ("success", "meta_framed", "leaked", "incoherent", "guessed")


def _cascade(semantic_leak: bool, meta_reference: bool, fluent: bool, guessing: bool = False) -> str:


    if semantic_leak:
        return "leaked"
    if not fluent:
        return "incoherent"
    if meta_reference:
        return "meta_framed"
    if guessing:
        return "guessed"
    return "success"


def derive_mode(j: JudgeOutput) -> str:

    return _cascade(j.semantic_leak, j.meta_reference, j.fluent, getattr(j, "guessing", False))


def mode_of(rec: Mapping) -> str:


    return _cascade(bool(rec["semantic_leak"]), bool(rec["meta_reference"]), bool(rec["fluent"]),
                    bool(rec.get("guessing", False)))


def _key(rec: Mapping, by: Sequence[str]) -> tuple:
    return tuple(rec.get(k) for k in by)


def _sorted_groups(groups: dict) -> list:
    return sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]])


def mode_distribution(records: Iterable[Mapping], by: Sequence[str] = ("concept_class", "prompt_id")):


    groups: dict[tuple, Counter] = {}
    for r in records:
        g = groups.setdefault(_key(r, by), Counter())
        g[mode_of(r)] += 1
        g["_n"] += 1
    rows = []
    for k, c in _sorted_groups(groups):
        n = c["_n"]
        row = dict(zip(by, k))
        row["n"] = n
        for m in MODES:
            row[m] = c.get(m, 0)
            row[f"{m}_frac"] = round(c.get(m, 0) / n, 4) if n else 0.0
        rows.append(row)
    return rows


def conditional_stats(records: Iterable[Mapping], by: Sequence[str] = ("concept_class", "prompt_id")):


    def blank() -> dict:
        return {"n": 0, "leak": 0, "meta": 0, "incoh": 0, "guess": 0,
                "meta_n": 0, "leak_meta": 0, "nometa_n": 0, "leak_nometa": 0}

    groups: dict[tuple, dict] = {}
    for r in records:
        g = groups.setdefault(_key(r, by), blank())
        leak, meta, fluent = bool(r["semantic_leak"]), bool(r["meta_reference"]), bool(r["fluent"])
        g["n"] += 1
        g["leak"] += leak
        g["meta"] += meta
        g["incoh"] += (not fluent)
        g["guess"] += bool(r.get("guessing", False))
        if meta:
            g["meta_n"] += 1
            g["leak_meta"] += leak
        else:
            g["nometa_n"] += 1
            g["leak_nometa"] += leak

    def frac(a: int, b: int):
        return round(a / b, 4) if b else None

    rows = []
    for k, g in _sorted_groups(groups):
        row = dict(zip(by, k))
        row.update({
            "n": g["n"],
            "p_leak": frac(g["leak"], g["n"]),
            "p_meta": frac(g["meta"], g["n"]),
            "p_incoherent": frac(g["incoh"], g["n"]),
            "p_guess": frac(g["guess"], g["n"]),
            "p_leak_given_meta": frac(g["leak_meta"], g["meta_n"]),
            "p_leak_given_no_meta": frac(g["leak_nometa"], g["nometa_n"]),
        })
        rows.append(row)
    return rows
