from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence


MODES: tuple[str, ...] = ("success", "meta_framed", "leaked", "incoherent")


def mode_of(rec: Mapping) -> str | None:


    c = rec.get("concept_score")
    if c in ("", None):
        return None
    c = int(c)
    if c == 0:
        return "success"
    f = rec.get("framing_score")
    if f in ("", None):
        return "leaked"          
    return "meta_framed" if int(f) == 2 else "leaked"


def _key(rec: Mapping, by: Sequence[str]) -> tuple:
    return tuple(rec.get(k) for k in by)


def rating_table(records: Iterable[Mapping], by: Sequence[str] = ("concept_class", "prompt_id")):

    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault(_key(r, by), []).append(r)

    def frac(a: int, b: int):
        return round(a / b, 4) if b else None

    rows = []
    for k, recs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        scored = [r for r in recs if r.get("concept_score") not in ("", None)]
        n = len(scored)
        leaks = [r for r in scored if int(r["concept_score"]) > 0]
        framed = [r for r in leaks if r.get("framing_score") not in ("", None)]
        fc = Counter(int(r["framing_score"]) for r in framed)
        modes = Counter(m for m in (mode_of(r) for r in scored) if m)
        row = {**dict(zip(by, k)), "n": n, "n_unparsed": len(recs) - n,
               "leak_rate": frac(len(leaks), n),
               "strong_leak": frac(sum(1 for r in scored if int(r["concept_score"]) == 2), n),
               "mean_concept": round(sum(int(r["concept_score"]) for r in scored) / n, 4) if n else None,


               "framed_n": len(framed),
               "framing_coverage": frac(len(framed), len(leaks)),
               "reasoned_leak": frac(fc.get(0, 0), len(framed)),
               "framed_leak": frac(fc.get(1, 0), len(framed)),
               "meta_leak": frac(fc.get(2, 0), len(framed))}
        for m in MODES:
            row[m] = modes.get(m, 0)
            row[f"{m}_frac"] = frac(modes.get(m, 0), n)
        rows.append(row)
    return rows
