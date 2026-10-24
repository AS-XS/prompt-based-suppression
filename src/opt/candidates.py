from __future__ import annotations

from src.opt.clause import best_wellformed, looks_like_meta, meta_reason, restates_instrument


def describe_search(compiled, found, seed, seed_score) -> dict:


    cands = getattr(compiled, "candidate_programs", []) or []
    clauses = {c["program"].signature.instructions for c in cands if isinstance(c, dict)
               and hasattr(getattr(c.get("program"), "signature", None), "instructions")}


    alt, alt_score, n_meta = best_wellformed(cands)
    if looks_like_meta(found):
        print(f"\n⚠️  the winning clause is META-TEXT — it {meta_reason(found)}.")
        print("    Prepended to the user turn this asks the model to DESIGN a prompt, so the")
        print("    score describes a malformed input, not a suppression wording.")
        if alt:
            print(f"    Best well-formed candidate scored {alt_score:.2f}: {alt[:120]}…")
        else:
            print("    NO candidate was well-formed — this run produced no usable prompt.")


    candidates = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        prog = c.get("program")
        txt = getattr(getattr(prog, "signature", None), "instructions", None)
        if not txt:
            continue
        demos = list(getattr(prog, "demos", []) or [])
        candidates.append({
            "clause": txt,


            "n_demos": len(demos),
            "demo_answers": [str(getattr(d, "response", "") or
                                 (d.get("response") if hasattr(d, "get") else ""))[:60]
                             for d in demos],
            "score": (round(float(c["score"]), 4) if c.get("score") is not None else None),
            "is_meta": looks_like_meta(txt),
            "restates_instrument": restates_instrument(txt, seed),
            "is_seed": txt == seed,
        })


    clean = [c for c in candidates
             if c["score"] is not None and not c["is_meta"]
             and not c["restates_instrument"] and not c["is_seed"]]
    best_clean = max(clean, key=lambda c: c["score"]) if clean else None
    if best_clean:
        print(f"\nbest candidate that is neither meta-text NOR a restatement of the instrument: "
              f"{best_clean['score']:.2f} (seed {seed_score:.2f})")
    else:
        print("\nNO candidate was both well-formed and free of instrument restatement.")

    search = {
        "candidates": candidates,
        "n_clean_candidates": len(clean),
        "best_clean_score": best_clean["score"] if best_clean else None,
        "best_clean_clause": best_clean["clause"] if best_clean else None,
        "n_candidates": len(cands),
        "found_is_meta": looks_like_meta(found),
        "meta_reason": meta_reason(found) if looks_like_meta(found) else "",
        "n_meta_candidates": n_meta,
        "best_wellformed_clause": alt,
        "best_wellformed_score": alt_score,
        "n_distinct_clauses": len(clauses),
        "seed_among_candidates": seed in clauses,
        "n_trials": len(getattr(compiled, "trial_logs", {}) or {}),
        "trial_scores": sorted({round(float(c["score"]), 4) for c in cands
                                if isinstance(c, dict) and c.get("score") is not None}),
        "best_trial_score": getattr(compiled, "score", None),


        "restates_instrument": restates_instrument(found, seed),
    }
    return search
