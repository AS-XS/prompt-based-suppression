#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts  # noqa: E402
from src.judge.settings import STAGES, judge_fingerprints  # noqa: E402
from src.paths import judgments_dir  # noqa: E402

RATING_STAGES = STAGES["rating"]


def coverage(rows, fingerprints: dict[str, str]):

    concept: dict[tuple, int | None] = {}
    at: dict[str, set] = defaultdict(set)
    other: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        key = (r.get("generation_id"), r.get("surface"))
        if r.get("concept_score") is not None:
            concept[key] = r["concept_score"]
        for stage in RATING_STAGES:
            if r.get(f"{stage}_score") is None:
                continue

            field = "judge_config_hash" if stage == "concept" else f"{stage}_config_hash"
            got = (r.get(field) or "")[:16]
            if got == fingerprints[stage]:
                at[stage].add(key)
            elif got:
                other[stage][got] += 1
    pairs = set(concept)
    eligible = {"concept": pairs, "mention": pairs,
                "framing": {k for k, v in concept.items() if v and v > 0}}
    return {s: (len(at[s] & eligible[s]), len(eligible[s]), dict(other[s])) for s in RATING_STAGES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--judge", default="Qwen3.6-27B")
    ap.add_argument("--quant", default="bitsandbytes")
    ap.add_argument("--require", nargs="*", choices=RATING_STAGES, default=[],
                    help='Require.')
    args = ap.parse_args()

    paths = sorted(glob.glob(str(judgments_dir(args.run) / f"{args.sample}*.jsonl")))
    if not paths:
        raise SystemExit(f"no judgment files for sample {args.sample!r} in "
                         f"{judgments_dir(args.run)} — has the judge run at all?")
    rows = [r for p in paths for r in _read_dicts(Path(p))]
    fp = judge_fingerprints("rating", args.judge, quant=args.quant)
    cov = coverage(rows, fp)

    print(f"{len(rows):,} record(s) over {len(paths)} file(s) · sample {args.sample}")
    short = []
    for stage in RATING_STAGES:
        n, d, others = cov[stage]
        pct = 100.0 * n / d if d else 0.0
        flag = "OK " if d and n == d else "!! "


        miss = f"   {d - n:,} MISSING" if d and n != d else ""
        print(f"  {flag}{stage:<8} {fp[stage]}  {n:,}/{d:,} = {pct:5.1f}%{miss}"
              + (f"   superseded: {others}" if others else ""))
        if not d or n != d:
            short.append(stage)
    if short:
        print(f"\nincomplete at the current rubric: {', '.join(short)}")
        print("  the judge resumes per (generation_id, surface) AND per rubric hash, so re-running")
        print("  the stage picks up exactly what is missing (COMMANDS §6k).")
    else:
        print("\nevery stage complete at the rubric the code computes now.")

    bad = [s for s in args.require if s in short]
    if bad:
        print(f"\nREQUIRED but incomplete: {', '.join(bad)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
