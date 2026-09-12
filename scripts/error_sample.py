#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts, write_csv  # noqa: E402
from src.analysis.accuracy import score_generations  # noqa: E402
from src.analysis.errors import (looks_like_refusal, outcome_table,  # noqa: E402
                                 pair_outcomes, stratified_sample)
from src.analysis.grouping import eval_subject  # noqa: E402
from src.analysis.surfaces import answer_of, explanation_of  # noqa: E402
from src.paths import analysis_dir, generations_glob, items_path  # noqa: E402

_SAMPLE_COLS = ["model", "think_state", "prompt_id", "concept_id", "mmlu_subject", "probe_type",
                "outcome", "baseline_pred", "pred", "parse_tier", "refusal",
                "mentions_concept", "explanation", "raw_output"]


def _write_csv(path: Path, rows, cols):
    write_csv(path, rows, cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp1")
    ap.add_argument("--items", default=None)
    ap.add_argument("--generations", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None, help='Models.')
    ap.add_argument("--per-cell", type=int, default=5,
                    help='Per cell.')
    ap.add_argument("--outcomes", nargs="+", default=None,
                    help='Outcomes.')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--png", action="store_true",
                    help='Png.')
    args = ap.parse_args()

    items_file = Path(args.items) if args.items else items_path(args.run)
    if not items_file.exists():
        raise SystemExit(f"no item set at {items_file}")
    paths = [f for pat in (args.generations or [generations_glob(args.run)])
             for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no generations for run {args.run!r}")

    idx = {}
    for it in _read_dicts(items_file):
        meta = it.get("meta") or {}
        idx[it["item_id"]] = {"answer_key": it.get("answer_key"), "format": meta.get("format"),
                              "mmlu_subject": eval_subject(it),
                              "probe_type": it.get("probe_type"),
                              "concept_id": it.get("concept_id")}

    gens = [g for p in paths for g in _read_dicts(Path(p))]
    if args.models:
        gens = [g for g in gens if g.get("model") in set(args.models)]
    raw = {(g.get("model"), g.get("think_state"), g.get("prompt_id"), g.get("item_id")):
           g.get("raw_output") for g in gens}

    scored = score_generations(gens, idx)
    if not scored:
        raise SystemExit("no MC generations scored")
    paired = pair_outcomes(scored)
    if not paired:
        raise SystemExit("no suppressed rows could be paired with a baseline — was the "
                         "none-baseline wave generated?")

    out = Path(args.out_dir) if args.out_dir else analysis_dir(args.run) / "errors"


    table = outcome_table(paired, by=("model", "think_state", "prompt_id", "probe_type"))
    _write_csv(out / "outcomes.csv", table,
               ["model", "think_state", "prompt_id", "probe_type", "n", "n_eligible",
                "undesired_rate", "held", "leaked_correct", "intact", "collateral", "unparsed",
                "already_wrong"])

    if args.png:
        from src.analysis.figures import outcome_png, outcome_summary_png, table_png

        outcome_summary_png(table, out / "outcomes_summary.png",
                            title=f"{args.run} — did suppression bite on target, or everywhere?")
        outcome_png(table, out / "outcomes.png",
                    title=f"{args.run} — what suppression actually did, per probe distance")
        table_png(table, out / "outcomes_table.png", title=f"{args.run} — error outcome mix",
                  note="undesired_rate = (leaked_correct + collateral + unparsed) / n_eligible. "
                       "n_eligible excludes already_wrong — rows the baseline never got right, "
                       "where there was nothing for suppression to remove.")

    sample = stratified_sample(paired, per_cell=args.per_cell, seed=args.seed, only=args.outcomes)
    for r in sample:
        text = raw.get((r["model"], r["think_state"], r["prompt_id"], r["item_id"])) or ""
        r["raw_output"] = " ".join(text.split())[:600]
        r["explanation"] = " ".join(explanation_of(text).split())[:400]
        r["refusal"] = looks_like_refusal(answer_of(text))


        r["mentions_concept"] = str(r.get("concept_id", "")).lower() in text.lower()
    _write_csv(out / "samples.csv", sample, _SAMPLE_COLS)

    md = [f"# {args.run} — error samples", "",
          f"{len(sample)} rows, seed {args.seed}, ≤{args.per_cell} per "
          f"(prompt × probe_type × outcome) cell. Outcomes are the model's suppressed answer "
          f"compared with **its own baseline on the same item**.", ""]
    for r in sample:
        md += [f"### {r['outcome']} · {r['model']} · {r['prompt_id']} · {r['probe_type']} · "
               f"suppressed={r['concept_id']} · tested={r.get('mmlu_subject')}",
               f"- baseline chose **{r['baseline_pred']}** (correct) → suppressed chose "
               f"**{r['pred']}** · parse tier `{r.get('parse_tier')}`"
               + ("  · **refusal-like**" if r["refusal"] else "")
               + ("  · **names the suppressed concept**" if r["mentions_concept"] else ""),
               f"- explanation: {r['explanation'] or '(none)'}", ""]
    (out / "samples.md").write_text("\n".join(md), encoding="utf-8")

    print(f"{len(paired)} paired rows over {len(paths)} file(s)")
    print(f"  outcome mix -> {out / 'outcomes.csv'}  ({len(table)} groups)")
    if args.png:
        print(f"  SUMMARY fig -> {out / 'outcomes_summary.png'}   <- the small one, start here")
        print(f"  full figure -> {out / 'outcomes.png'}  +  outcomes_table.png")
    print(f"  samples     -> {out / 'samples.csv'}  +  samples.md  ({len(sample)} rows)")
    c = Counter(r["outcome"] for r in paired)
    tot = sum(c.values()) - c["already_wrong"]
    print(f"\n  outcome mix (all models pooled; already_wrong excluded from rates):")
    for k in ("held", "leaked_correct", "intact", "collateral", "unparsed", "already_wrong"):
        pct = f"{100 * c[k] / tot:5.1f}%" if tot and k != "already_wrong" else "     "
        print(f"    {k:16} {c[k]:7}  {pct}")
    if sample:
        ref = sum(1 for r in sample if r["refusal"])
        men = sum(1 for r in sample if r["mentions_concept"])
        print(f"\n  in the sample: {ref}/{len(sample)} look like refusals, "
              f"{men}/{len(sample)} name the suppressed concept")
    print(f"\n  READ: {out / 'samples.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
