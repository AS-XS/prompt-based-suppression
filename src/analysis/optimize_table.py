from __future__ import annotations

from math import comb


def discordance(seed_items: dict, found_items: dict) -> tuple[int, int, int]:

    shared = sorted(set(seed_items) & set(found_items))
    seed_only = sum(1 for i in shared if seed_items[i] and not found_items[i])
    found_only = sum(1 for i in shared if found_items[i] and not seed_items[i])
    return seed_only, found_only, len(shared)


def mcnemar_exact(seed_only: int, found_only: int) -> float | None:

    n = seed_only + found_only
    if n == 0:
        return None
    k = min(seed_only, found_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def verdict(seed_only: int, found_only: int, p: float | None) -> str:


    if p is None:
        return "identical on every item — the search changed nothing"
    if p >= 0.05:
        return (f"no reliable difference (p={p:.3f}) — {found_only} item(s) gained, "
                f"{seed_only} lost")
    return (f"{'found BEATS seed' if found_only > seed_only else 'found is WORSE than seed'} "
            f"(p={p:.3f}, {found_only} gained / {seed_only} lost)")


def summarise(run: dict) -> dict:

    seed_only, found_only, n = discordance(run.get("seed_items") or {}, run.get("found_items") or {})
    p = mcnemar_exact(seed_only, found_only)
    return {
        "model": run.get("model"),
        "objective": run.get("objective"),
        "concept": ", ".join(run.get("concepts") or []) or run.get("concept"),
        "think_state": run.get("think_state"),
        "seed_prompt": run.get("seed_prompt"),
        "n_dev": run.get("n_dev"),
        "seed_score": run.get("seed_score"),
        "found_score": run.get("found_score"),
        "delta": (None if run.get("found_score") is None or run.get("seed_score") is None
                  else round(run["found_score"] - run["seed_score"], 2)),
        "gained": found_only,
        "lost": seed_only,
        "n_paired": n,
        "p_value": p,
        "verdict": verdict(seed_only, found_only, p),
        "dry_run": bool(run.get("dry_run")),
        "search": run.get("search") or {},
    }


def search_note(search: dict, found_is_seed: bool) -> str:


    if not search:
        return ("_No search trace recorded — this run predates the instrumentation, so "
                "**`found == seed` cannot be read as \"the search tried and failed\"** here._"
                if found_is_seed else "")
    n_c = search.get("n_candidates") or 0
    n_d = search.get("n_distinct_clauses") or 0
    n_t = search.get("n_trials") or 0
    scores = search.get("trial_scores") or []
    spread = (f"scores {min(scores):.2f}–{max(scores):.2f}" if len(scores) > 1
              else (f"score {scores[0]:.2f}" if scores else "no scored candidates"))
    if n_c <= 1 or n_d <= 1:
        return (f"⚠️ **The search barely explored**: {n_c} candidate(s), {n_d} distinct clause(s) over "
                f"{n_t} trial(s). Do not read this as \"a search could not beat the hand prompt\".")
    body = (f"The search evaluated **{n_d} distinct clauses** across {n_c} candidates and {n_t} "
            f"trials ({spread}).")
    if found_is_seed:
        body += (" It returned the **hand-written seed**, so none of them beat it — which is the "
                 "point of running a search at all.")
    return body


_ARM = __import__("re").compile(r".*-(forget|neighbor|utility)-[0-9a-f]+$")


def per_arm(run: dict) -> list[dict]:


    seed, found = run.get("seed_items") or {}, run.get("found_items") or {}
    arms: dict[str, list[int]] = {}
    for k, v in seed.items():
        m = _ARM.match(k)
        a = arms.setdefault(m.group(1) if m else "?", [0, 0, 0])
        a[0] += 1; a[1] += int(v)
    for k, v in found.items():
        m = _ARM.match(k)
        if (m.group(1) if m else "?") in arms:
            arms[m.group(1) if m else "?"][2] += int(v)
    out = []
    for a in sorted(arms):
        n, sc, fo = arms[a]
        out.append({"arm": a, "n": n,
                    "seed_pct": round(sc / n * 100, 1) if n else None,
                    "found_pct": round(fo / n * 100, 1) if n else None,
                    "means": ("scored 1 = answered WRONG (suppression worked)" if a == "forget"
                              else "scored 1 = still CORRECT (no collateral)")})
    return out
