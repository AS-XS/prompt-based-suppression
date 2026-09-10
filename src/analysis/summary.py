from __future__ import annotations

import statistics as st

from src.analysis.report import load


_PROMPT_ORDER = ("world-frame", "ignorance-frame", "no-meta")


def _prompts(rows) -> list[str]:

    found = {c[: -len("_drop")] for r in rows for c in r if str(c).endswith("_drop")}
    known = [p for p in _PROMPT_ORDER if p in found]
    return known + sorted(found - set(known))


_CLEAN = {"off", "na"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def cross_model_rows(analysis_path) -> list[dict]:

    delta = load(analysis_path / "accuracy_delta.csv")          
    lens = load(analysis_path / "answer_lengths.csv")           
    mx = load(analysis_path / "subject_matrix.csv")             

    models = sorted({r["model"] for r in delta}) if delta else []


    lgen = {(r["model"], r["think_state"], r["prompt_id"]): r
            for r in lens
            if r.get("concept_class") == "general" and r.get("probe_type") == "forget"
            and r.get("format") in (None, "", "mc")}

    out = []
    for model in models:
        states = sorted({r["think_state"] for r in delta if r["model"] == model})
        for ts in states:
            drows = [r for r in delta if r["model"] == model and r["think_state"] == ts]
            base_len = lgen.get((model, ts, "none-baseline"), {})

            nb = sum(int(r.get("n_baseline") or 0) for r in drows)
            base_acc = (sum(float(r["baseline_acc"]) * int(r["n_baseline"]) for r in drows
                            if r.get("baseline_acc") not in ("", None)) / nb) if nb else None
            for p in _prompts(delta):
                dk = f"{p}_drop"
                if not drows or dk not in drows[0]:
                    continue
                drop = _mean([float(r[dk]) for r in drows if r.get(dk) not in ("", None)])
                lp = lgen.get((model, ts, p), {})
                bmed = int(base_len.get("median_chars") or 0)
                smed = int(lp.get("median_chars") or 0)
                fold = (bmed / smed) if smed else (bmed if bmed else None)

                mrows = [r for r in mx if r["model"] == model and r["think_state"] == ts]
                diag = _mean([float(r[dk]) for r in mrows
                              if r["concept_id"] == r["tested_concept"] and r.get(dk) not in ("", None)])
                off = _mean([float(r[dk]) for r in mrows
                             if r["concept_id"] != r["tested_concept"] and r.get(dk) not in ("", None)])
                gap = (diag - off) if (diag is not None and off is not None) else None


                supp_acc = (base_acc - drop) if (base_acc is not None and drop is not None) else None
                out.append({
                    "model": model,
                    "thinking": "off" if ts in _CLEAN else "on",
                    "prompt type": p,
                    "items": nb,
                    "acc: normal": _pct(base_acc),
                    "acc: suppressed": _pct(supp_acc),
                    "acc lost (pp)": _pp(drop),
                    "answer len: normal": bmed or "",
                    "answer len: suppressed": smed or "",
                    "shrinkage": (f"{fold:.0f}x" if fold else ""),
                    "% one-letter": _pct(lp.get("bare_pct")),


                    "blast radius": ("n/a — think-on acc confounded" if ts not in _CLEAN else
                                     "" if gap is None else
                                     "~none (no effect)" if (drop is not None and abs(drop) < 0.02) else
                                     "on-target only" if gap > 0.02 else "everywhere"),
                })
    return out


LEGEND = (
    "HOW TO READ — one row per (model, thinking on/off, prompt type). "
    "Clean arm for accuracy = thinking OFF; thinking-ON accuracy can be distorted because the model "
    "runs out of tokens mid-reasoning, so on those rows trust the shrinkage, not the accuracy drop OR the blast radius (both are accuracy-based). "
    "COLUMNS: acc:normal = % correct on the SUPPRESSED concept's OWN multiple-choice questions with no "
    "suppression, averaged over that concept's MMLU subjects and all 6 concepts (chance = 25%) · "
    "acc:suppressed = the same after the suppression prompt · acc lost = normal − suppressed in pp, "
    "so READ THE SIGN AS: +N = suppression made it N points WORSE, −N = it did N points BETTER (that "
    "only happens on the confounded thinking-on rows). The effect on OTHER subjects is the "
    "blast-radius column, not this one · "
    "answer len normal→suppressed = median answer length in characters · "
    "shrinkage = how many times shorter the answer got (35x means a 1183-char answer became ~34) · "
    "% one-letter = share of answers that collapsed to a bare letter like \"C\" · blast radius = does "
    "suppressing concept X hurt only X (on-target only) or every subject (everywhere)? "
    "THE POINT: a huge shrinkage next to a tiny acc. drop = the behaviour collapsed but accuracy is "
    "blind to it. pp = percentage points."
)


def _pp(frac):
    return "" if frac is None else f"{frac*100:+.1f}"


def _pct(frac):
    if frac in ("", None, False):
        return ""
    try:
        return f"{float(frac)*100:.0f}%"
    except (TypeError, ValueError):
        return ""
