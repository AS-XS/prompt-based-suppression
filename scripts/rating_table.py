#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts  # noqa: E402
from src.analysis.ratings import MODES, rating_table  # noqa: E402
from src.paths import analysis_dir, judgments_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp1")
    ap.add_argument("--judgments", nargs="+", default=None, help='Judgments.')
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--by", nargs="+", default=["concept_class", "prompt_id"],
                    help='By.')
    ap.add_argument("--surface", default=None,
                    help='Surface.')
    args = ap.parse_args()

    globs = args.judgments or [str(judgments_dir(args.run) / "*.jsonl")]
    paths = [f for pat in globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no judgment files matched: {globs}\n"
                         f"  run the 0-2 judge first: bash scripts/judge_all.sh")


    from src.analysis.ratings import completeness
    best: dict[tuple, dict] = {}
    dupes = 0
    for p in paths:
        for r in _read_dicts(Path(p)):
            key = (r.get("generation_id"), r.get("surface"))
            if key in best:
                dupes += 1
                if completeness(r) < completeness(best[key]):
                    continue
            best[key] = r
    records = list(best.values())
    if args.surface:
        records = [r for r in records if r.get("surface") == args.surface]
    if not records:
        raise SystemExit("no judgment rows (wrong --surface?)")

    rows = rating_table(records, by=args.by)
    cols = (list(args.by) + ["n", "n_unparsed", "leak_rate", "strong_leak", "mean_concept",
                             "n_contradiction", "contra_concept_framing", "contra_framing_mention",
                             "framed_n", "framing_coverage", "used_share",
                             "named_only_share", "framing1_n", "mention_n", "mention_coverage",
                             "mention_rate", "exact_mention", "direct_leak", "inadvertent_leak",
                             "p_used_given_mentioned"]
            + [c for m in MODES for c in (m, f"{m}_frac")])

    out = Path(args.out_dir) if args.out_dir else analysis_dir(args.run)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "rating_distribution.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(records)} judgments over {len(paths)} file(s), grouped by {args.by}"
          + (f"  ({dupes} duplicate row(s) across files ignored)" if dupes else ""))
    print(f"  0-2 frame table -> {dest}  ({len(rows)} groups)")
    unparsed = sum(r["n_unparsed"] for r in rows)
    if unparsed:
        print(f"  ⚠️  {unparsed} row(s) had no concept_score — the judge's "
              f'"Rating: [[n]]" line did not parse')


    leaks = sum(int(round((r["leak_rate"] or 0) * r["n"])) for r in rows)
    framed = sum(r["framed_n"] for r in rows)
    if leaks and framed < leaks:
        pct = 100 * framed / leaks
        print(f"  ⚠️  FRAMING INCOMPLETE — {framed}/{leaks} leaked rows have a framing score "
              f"({pct:.0f}%).")
        print(f"      leak_rate / strong_leak / mean_concept are FINAL and safe to read.")
        print(f"      used|named_only_share and the mode split (leaked vs meta_framed) are "
              f"PROVISIONAL:")
        print(f"      an unframed leak counts as `leaked`, so `meta_framed` will grow as stage 2 runs.")
        print(f"      finish it with:  python scripts/judge.py ... --stage framing")
    scored = sum(r["n"] for r in rows)
    mentions = sum(r["mention_n"] for r in rows)
    if scored and mentions < scored:
        print(f"  ⚠️  MENTION INCOMPLETE — {mentions}/{scored} rows have a mention score. "
              f"meta_framed vs success, direct vs inadvertent and p_used_given_mentioned are "
              f"PROVISIONAL until it is 100%:  python scripts/judge.py ... --stage mention")
    contra = sum(r["n_contradiction"] for r in rows)
    if contra:
        print(f"  ⚠️  {contra} row(s) where two stages contradict each other (concept 2 & framing 2, "
              f"or framing 2 & mention 0) — refused: no mode (n_contradiction, contra_*).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
