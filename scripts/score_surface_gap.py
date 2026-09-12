#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts  # noqa: E402
from src.analysis.surface_gap import surface_gap_table  # noqa: E402
from src.data.terms import has_vocabulary  # noqa: E402
from src.paths import analysis_dir, generations_glob  # noqa: E402
from src.provenance import select_one_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help='Run.')
    ap.add_argument("--generations", nargs="+", default=None, help='Generations.')
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config-hash", default=None,
                    help='Config hash.')
    ap.add_argument("--by", nargs="+",
                    default=["model", "think_state", "prompt_id", "concept_class", "concept_id",
                             "probe_type"],
                    help='By.')
    args = ap.parse_args()

    if not args.run and not args.generations:
        raise SystemExit("pass --run exp1|round1 (or --generations explicitly)")
    gen_globs = args.generations or [generations_glob(args.run)]
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no generation files matched: {gen_globs}")

    gens = [g for p in paths for g in _read_dicts(Path(p))]
    gens = select_one_config(gens, args.config_hash, what="generations")


    missing = sorted({g.get("concept_id") for g in gens
                      if g.get("concept_class") == "general" and not has_vocabulary(g.get("concept_id") or "")})
    if missing:
        raise SystemExit(
            f"no surface vocabulary for {missing} — this scorer would report 0.0% leakage for "
            f"them on every row, which is not a result.\nAdd them to src/data/terms.py"
            f"::CONCEPT_TERMS (the DOMAIN's vocabulary, not the suppression prompt's wording).")

    rows = surface_gap_table(gens, by=tuple(args.by))
    if not rows:
        raise SystemExit("no generations to score")

    out = Path(args.out_dir) if args.out_dir else analysis_dir(args.run)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "surface_gap.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:


        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"scored {len(gens)} generations over {len(paths)} file(s)")
    print(f"  surface gap -> {dest}  ({len(rows)} groups)")
    paired = sum(r["n_paired"] for r in rows)
    if not paired:
        print("  ⚠️  no rows carried BOTH surfaces — the paired columns are empty. Only think-ON "
              "rows have a trace, so this is expected if you scored a think-off-only sweep.")
    else:
        t = sum(r["trace_only"] for r in rows)
        e = sum(r["explanation_only"] for r in rows)
        print(f"  paired rows {paired}: leaked in TRACE only {t}, in EXPLANATION only {e}")


        tn = sum(r["trace_only_novel"] or 0 for r in rows)
        en = sum(r["explanation_only_novel"] or 0 for r in rows)
        pn = sum(r["n_paired_novel"] or 0 for r in rows)
        unmeasurable = sum(1 for r in rows if r["n_paired_novel"] is None)
        print(f"  excluding the prompt's own words: {pn} paired, TRACE only {tn}, "
              f"EXPLANATION only {en}" + (f"  ({unmeasurable} group(s) unmeasurable)"
                                          if unmeasurable else ""))
        print("  (a lexical proxy — a trace hit can be the model restating the instruction; "
              "the judge's semantic_leak is the instrument)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
