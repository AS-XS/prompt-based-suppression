from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence


MODES: tuple[str, ...] = ("success", "meta_framed", "leaked", "incoherent")
CELLS: tuple[str, ...] = ("used_named", "used_unnamed", "incoherent", "unused_named",
                          "unused_unnamed")
_MODE_OF_CELL = {"used_named": "leaked", "used_unnamed": "leaked", "incoherent": "incoherent",
                 "unused_named": "meta_framed", "unused_unnamed": "success"}


SCORES = ("concept_score", "framing_score", "mention_score")


def _score(rec: Mapping, field: str) -> int | None:
    v = rec.get(field)
    return None if v in ("", None) else int(v)


def completeness(rec: Mapping) -> int:


    return sum(rec.get(k) not in ("", None) for k in SCORES)


def best_records(dicts: Iterable[Mapping]) -> tuple[dict, int]:


    best: dict[tuple, dict] = {}
    dupes = 0
    for d in dicts:
        k = (d.get("generation_id"), d.get("surface"))
        if k in best:
            dupes += 1
            if completeness(d) < completeness(best[k]):
                continue
        best[k] = dict(d)
    return best, dupes


def present(rec: Mapping) -> bool:
    c = _score(rec, "concept_score")
    return c is not None and c > 0


def used(rec: Mapping) -> bool:
    return present(rec) and _score(rec, "framing_score") != 2


def mentioned(rec: Mapping) -> bool:
    m = _score(rec, "mention_score")
    if m is not None:
        return m > 0
    return _score(rec, "framing_score") == 2      


def _flag(v) -> bool:
    return v in (True, 1, "1", "True", "true")


def cell_of(rec: Mapping) -> str | None:


    if _score(rec, "concept_score") is None:
        return None
    if used(rec):
        return "used_named" if mentioned(rec) else "used_unnamed"
    if _flag(rec.get("incoherent")):
        return "incoherent"
    return "unused_named" if mentioned(rec) else "unused_unnamed"


def mode_of(rec: Mapping) -> str | None:

    cell = cell_of(rec)
    return None if cell is None else _MODE_OF_CELL[cell]


def _key(rec: Mapping, by: Sequence[str]) -> tuple:
    return tuple(rec.get(k) for k in by)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:

    if not n:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def rating_table(records: Iterable[Mapping], by: Sequence[str] = ("concept_class", "prompt_id")):


    groups: dict[tuple, list] = {}
    for r in records:
        groups.setdefault(_key(r, by), []).append(r)

    def frac(a: int, b: int):
        return round(a / b, 4) if b else None

    rows = []
    for k, recs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        scored = [r for r in recs if _score(r, "concept_score") is not None]
        n = len(scored)
        leaks = [r for r in scored if present(r)]
        n_used = sum(1 for r in scored if used(r))
        framed = [r for r in leaks if _score(r, "framing_score") is not None]
        fc = Counter(_score(r, "framing_score") for r in framed)
        with_mention = [r for r in scored if _score(r, "mention_score") is not None]
        mc = Counter(_score(r, "mention_score") for r in with_mention)
        cells = Counter(cell_of(r) for r in scored)
        modes = Counter(mode_of(r) for r in scored)

        named = [r for r in with_mention if mentioned(r)]
        used_named = sum(1 for r in named if used(r))
        used_lo, used_hi = wilson(n_used, n)
        rq3_lo, rq3_hi = wilson(used_named, len(named))
        row = {**dict(zip(by, k)), "n": n, "n_unparsed": len(recs) - n,
               "leak_rate": frac(len(leaks), n),
               "strong_leak": frac(sum(1 for r in scored if _score(r, "concept_score") == 2), n),
               "mean_concept": (round(sum(_score(r, "concept_score") for r in scored) / n, 4)
                                if n else None),
               "used_rate": frac(n_used, n), "used_lo": used_lo, "used_hi": used_hi,


               "framed_n": len(framed),
               "framing_coverage": frac(len(framed), len(leaks)),
               "used_share": frac(fc.get(0, 0) + fc.get(1, 0), len(framed)),
               "named_only_share": frac(fc.get(2, 0), len(framed)),
               "framing1_n": fc.get(1, 0),          
               "mention_n": len(with_mention),
               "mention_coverage": frac(len(with_mention), n),
               "mention_rate": frac(mc.get(1, 0) + mc.get(2, 0), len(with_mention)),
               "exact_mention": frac(mc.get(2, 0), len(with_mention)),
               "n_mentioned": len(named), "n_used_mentioned": used_named,
               "p_used_given_mentioned": frac(used_named, len(named)),
               "p_used_given_mentioned_lo": rq3_lo, "p_used_given_mentioned_hi": rq3_hi}
        for c in CELLS:
            row[c] = cells.get(c, 0)
            row[f"{c}_frac"] = frac(cells.get(c, 0), n)
        for m in MODES:
            row[m] = modes.get(m, 0)
            row[f"{m}_frac"] = frac(modes.get(m, 0), n)
        rows.append(row)
    return rows
