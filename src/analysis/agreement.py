from __future__ import annotations

from typing import Mapping, Sequence

from src.analysis.judge_sample import kappa
from src.analysis.ratings import used

AXES = ("present", "used", "named")
_YES, _NO = {"y", "yes", "1", "true"}, {"n", "no", "0", "false"}


def parse_labels(rows: Sequence[Mapping]) -> dict[int, dict]:


    out, errors = {}, []
    for r in rows:
        n = int(r["n"])
        present, used, named = ((r.get(k) or "").strip().lower() for k in AXES)
        if not (present or used or named):
            continue
        if present not in ("0", "1", "2") or named not in ("0", "1", "2"):
            errors.append(f"row {n}: present and named must be 0, 1 or 2")
            continue
        if present == "0":
            if used in _YES:
                errors.append(f"row {n}: used is y but present is 0")
                continue
            flag = False
        elif used in _YES or used in _NO:
            flag = used in _YES
        else:
            errors.append(f"row {n}: present is {present}, so used must be y or n")
            continue
        out[n] = {"present": int(present), "used": flag, "named": int(named)}
    if errors:
        raise ValueError("label sheet has malformed rows:\n  " + "\n  ".join(errors))
    return out


def judge_labels(key_rows: Sequence[Mapping]) -> dict[int, dict]:


    out = {}
    for r in key_rows:
        c, f, m = (r.get(k) for k in ("concept_score", "framing_score", "mention_score"))
        if c in ("", None) or m in ("", None) or (int(c) > 0 and f in ("", None)):
            continue
        rec = {"concept_score": c, "framing_score": f, "mention_score": m}
        out[int(r["n"])] = {"present": int(c), "used": used(rec), "named": int(m)}
    return out


def _binary(v, axis: str) -> bool:
    return bool(v) if axis == "used" else v > 0


def compare(a: Mapping[int, dict], b: Mapping[int, dict]) -> list[dict]:


    common = sorted(set(a) & set(b))
    rows = []
    for axis in AXES:
        n11 = n10 = n01 = n00 = exact = 0
        for i in common:
            x, y = _binary(a[i][axis], axis), _binary(b[i][axis], axis)
            n11 += x and y
            n10 += x and not y
            n01 += y and not x
            n00 += not x and not y
            exact += a[i][axis] == b[i][axis]
        n = len(common)
        rows.append({"axis": axis, "n": n, "both": n11, "a_only": n10, "b_only": n01,
                     "neither": n00, "agreement": round((n11 + n00) / n, 4) if n else None,
                     "kappa": kappa(n11, n10, n01, n00),
                     "exact_0_1_2": (round(exact / n, 4) if n and axis != "used" else None)})
    return rows
