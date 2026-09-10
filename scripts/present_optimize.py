#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.optimize_table import per_arm, search_note, summarise  # noqa: E402
from src.opt.clause import looks_like_meta, meta_reason, restates_instrument  # noqa: E402
from src.paths import optimize_dir, presentation_dir  # noqa: E402

_HEADER = """# {run} — the DSPy arm

Each row compares a **searched** suppression clause against the **hand-written** rung it started
from, on the same held-out `val` items. The two arms are prompt-only and the model is frozen in
both; only the author of the clause differs.

> **Read Δ with `gained`/`lost`, never alone.** Seed and found are scored on the same items, so the
> honest test is paired (McNemar's exact). With {n_dev} dev items **one item is one point** — a
> "+3.00" is three questions changing answer.

| objective | what it rewards | the claim it supports |
|---|---|---|
| **A** | `held` on forget, `intact` off-target | unlearning-comparable — reads against RMU 84% / ELM 90% |
| **B** | forget answered **wrongly**, retain unconstrained | behavioural suppression — a state weight editing cannot reach |

Never collapse them: below-chance accuracy is *evidence the knowledge is intact*, so B's number is
not an unlearning result.
"""

_FOOTER = """
---

## What is not yet shown here

A found clause is a **candidate selected on `val`**, not a result. To make it one: register it as a
`prompt_id`, sweep it through the ordinary pipeline, and score it on `test`. Until then this page
says only which clause won a selection contest, on the half of the data set aside for running it.
"""


def _rows(paths: list[Path]) -> tuple[list[dict], list[dict], list[tuple]]:
    real, dry, bad = [], [], []
    for p in sorted(paths):


        try:
            raw = p.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except json.JSONDecodeError:
            bad.append((p, "empty (0 bytes) — the job died before writing"
                           if not raw.strip() else "not valid JSON — truncated mid-write"))
            continue
        except OSError as e:
            bad.append((p, f"unreadable: {e}"))
            continue
        if str(payload.get("status", "")).startswith("seed-only"):


            bad.append((p, f"seed-only bank (seed {payload.get('seed_score')}) — the search did "
                           f"not finish; re-run it, do not read it as a null"))
            continue
        row = summarise(payload)
        row["_path"] = p
        (dry if row["dry_run"] else real).append(row)
    return real, dry, bad


def _table(rows: list[dict]) -> str:
    head = ("| model | obj | concept | think | seed | found | Δ | gained | lost | p | verdict |\n"
            "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n")
    out = []
    for r in rows:
        p = "—" if r["p_value"] is None else f"{r['p_value']:.3f}"
        d = "—" if r["delta"] is None else f"{r['delta']:+.2f}"
        out.append(f"| `{r['model']}` | **{r['objective']}** | {r['concept']} | {r['think_state']} "
                   f"| {r['seed_score']:.2f} | {r['found_score']:.2f} | {d} "
                   f"| {r['gained']} | {r['lost']} | {p} | {r['verdict']} |")
    return head + "\n".join(out) + "\n"


def _clauses(rows: list[dict]) -> str:
    out = []
    for r in rows:
        run = json.loads(r["_path"].read_text(encoding="utf-8"))
        seed, found = run["seed_clause"], run["found_clause"]
        out.append(f"\n### {r['model']} · objective {r['objective']} · {r['concept']}\n")
        out.append(f"**Seed** — the hand-written rung `{r['seed_prompt']}`:\n\n> {seed}\n")
        same = found.strip() == seed.strip()
        if same:
            out.append("\n**Found:** *the optimiser returned the seed unchanged.*\n")
        else:
            out.append(f"\n**Found** — what the optimiser wrote:\n\n> {found}\n")
        note = search_note(r.get("search") or {}, same)
        if note:
            out.append(f"\n{note}\n")
        sr = run.get("search") or {}

        is_meta = sr.get("found_is_meta")
        if is_meta is None:
            is_meta = looks_like_meta(found)
            sr = {**sr, "meta_reason": meta_reason(found)}
        if is_meta:
            out.append(f"\n> ⚠️ **This clause is proposer META-TEXT** — it {sr.get('meta_reason','')}. "
                       f"Prepended to the user turn it asks the model to *design a prompt*, so the "
                       f"score describes a malformed input rather than a suppression wording. "
                       f"{sr.get('n_meta_candidates', 0)} of {sr.get('n_candidates', 0)} candidates "
                       f"were meta-text.\n")
            if sr.get("best_wellformed_clause"):
                out.append(f">\n> Best **well-formed** candidate "
                           f"(scored {sr.get('best_wellformed_score')}):\n>\n"
                           f"> {sr['best_wellformed_clause']}\n")
        if restates_instrument(found, seed):
            out.append("\n> ⚠️ **This clause repeats the answer-format instruction** that the "
                       "rendered question already carries, so the model reads the format twice. "
                       "That difference is not a suppression mechanism — do not attribute the whole "
                       "gain to wording.\n")
        arms = per_arm(run)
        if len(arms) > 1:
            out.append("\n| arm | n | seed | found | what a 1 means |\n|---|---:|---:|---:|---|\n")
            for a in arms:
                out.append(f"| {a['arm']} | {a['n']} | {a['seed_pct']}% | {a['found_pct']}% "
                           f"| {a['means']} |\n")

    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dspy-wmdp1", help='Run.')
    ap.add_argument("--optimize-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-png", action="store_true",
                    help='No png.')
    args = ap.parse_args()

    src = Path(args.optimize_dir) if args.optimize_dir else optimize_dir(args.run)
    paths = sorted(src.glob("*.json"))
    if not paths:
        raise SystemExit(f"no optimiser runs in {src} — has the arm run? "
                         f"(a dry run writes `<model>__<objective>.dryrun.json`)")

    real, dry, bad = _rows(paths)
    for path, why in bad:
        print(f"  ⚠️  SKIPPED {path.name}: {why}")
    if bad and not real and not dry:
        raise SystemExit(
            f"every result file in {paths[0].parent} is unreadable — nothing to report. "
            f"Check the job logs; a 0-byte file means the job died before writing.")
    n_dev = (real or dry)[0]["n_dev"]
    parts = [_HEADER.format(run=args.run, n_dev=n_dev)]
    if real:
        parts += ["\n## Scoreboard\n\n", _table(real), "\n## The clauses\n", _clauses(real)]
    else:
        parts += ["\n> **No completed optimiser runs yet** — only dry runs, which evaluate the seed "
                  "clause and stop. Nothing below is a comparison.\n"]
    if dry:
        parts += ["\n## Dry runs (wiring checks, not comparisons)\n\n", _table(dry)]
    parts.append(_FOOTER)

    out = Path(args.out) if args.out else presentation_dir(args.run) / "optimize.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(parts), encoding="utf-8")


    if real and not args.no_png:


        try:
            from src.analysis.figures import table_png

            cols = ["model", "objective", "concept", "seed_score", "found_score", "delta",
                    "gained", "lost", "p_value", "verdict"]
            png_rows = [{c: ("—" if r[c] is None else
                             f"{r[c]:.3f}" if c == "p_value" else
                             f"{r[c]:+.2f}" if c == "delta" else
                             f"{r[c]:.2f}" if c.endswith("_score") else r[c])
                         for c in cols} for r in real]
            png = table_png(png_rows, out.with_suffix(".png"),
                            title=f"{args.run} — searched clause vs hand-written seed",
                            note=(f"Paired on the same {n_dev} dev items (McNemar exact). "
                                  f"One item = {100 / n_dev:.1f} points, so read gained/lost and p, "
                                  f"not the delta alone. A found clause is a candidate selected on "
                                  f"val — not a result until swept and scored on test."))
            if png:
                print(f"  png -> {png}")
        except (ImportError, SystemExit):
            print("  (no PNG — matplotlib missing; pip install -e '.[analysis]'. "
                  "The markdown page above is complete either way.)")

    print(f"wrote {out}  ({len(real)} run(s), {len(dry)} dry)")
    for r in real:
        print(f"  {r['model']} obj {r['objective']}: seed {r['seed_score']:.2f} -> "
              f"found {r['found_score']:.2f}   {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
