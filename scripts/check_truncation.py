#!/usr/bin/env python


from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.repetition import LOOP_AT, SHINGLE, TAIL, loop_score  # noqa: E402,F401


def _scan(run: str, model: str, wave: str, show: int):

    stats: dict[tuple[str, str], collections.Counter] = collections.defaultdict(collections.Counter)
    lens: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    samples: list[tuple] = []
    files = sorted(glob.glob(f"results/{run}/generations/{model}__{wave}.jsonl"))
    if not files:
        print(f"no generations matching results/{run}/generations/{model}__{wave}.jsonl")
        return None, None, None
    for f in files:
        for line in open(f):
            r = json.loads(line)
            k = (r["model"], r.get("think_state") or "na")
            c = stats[k]
            c["n"] += 1
            if r.get("completion_tokens") is not None:
                lens[k].append(r["completion_tokens"])
            if r.get("finish_reason") == "stop":
                continue
            c["trunc"] += 1
            sc = loop_score(r.get("raw_output") or "")
            if sc > LOOP_AT:
                c["loop"] += 1
                if len(samples) < show:
                    samples.append((r["model"], r["prompt_id"], r["item_id"], sc,
                                    (r.get("raw_output") or "")[-400:]))
    return stats, lens, samples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="*")
    ap.add_argument("--wave", default="*",
                    help='Wave.')
    ap.add_argument("--compare", metavar="RUN",
                    help='Compare.')
    ap.add_argument("--show", type=int, default=0, metavar="N",
                    help='Show.')
    args = ap.parse_args()
    if args.compare and args.wave == "*":
        ap.error("--compare needs --wave: comparing one run's every-wave average against another's "
                 "is what set this gate's threshold wrong in the first place")

    stats, lens, samples = _scan(args.run, args.model, args.wave, args.show)
    if stats is None:
        return 1
    ref = _scan(args.compare, args.model, args.wave, 0)[:2] if args.compare else (None, None)

    rstats, rlens = ref
    head = (f"{'model':<26s}{'think':>6s}{'n':>8s}{'p50 tok':>9s}{'TRUNCATED':>11s}"
            f"{'…looping':>10s}{'…long':>8s}")
    if rstats is not None:
        head += f"{'  vs ' + args.compare:>16s}"
    print(head)


    partial = {m for m, _ in stats}
    partial = {m for m in partial
               if len({c["n"] for (mm, _), c in stats.items() if mm == m}) > 1}

    verdicts = []
    for k in sorted(stats):
        c, v = stats[k], sorted(lens[k])
        tr = c["trunc"]
        p50 = v[len(v) // 2] if v else 0
        at_cap = bool(v) and p50 >= max(v)


        split = (f"{c['loop'] / tr:>10.1%}{(tr - c['loop']) / tr:>8.1%}" if tr
                 else f"{'—':>10s}{'—':>8s}")
        line = (f"{k[0]:<26s}{k[1]:>6s}{c['n']:>8d}{p50:>9d}{c['trunc'] / c['n']:>10.1%}"
                f"{split}")
        if rstats is not None:
            rc = rstats.get(k)
            if rc and rc["n"]:
                was = rc["trunc"] / rc["n"]
                now = c["trunc"] / c["n"]
                line += (f"{was:>10.1%} ->"
                         + ("  same" if abs(now - was) < 1e-9 else
                            "  better" if now < was else "  WORSE"))
                verdicts.append((k, was, now, at_cap))
            else:
                line += f"{'(absent)':>16s}"
        if at_cap:
            line += "   <-- p50 IS THE CAP: this model does not terminate"
        if k[0] in partial:
            line += "   ⚠ PARTIAL"
        print(line)

    print(f"\n  …looping / …long are shares OF THE TRUNCATED ones, so they sum to 100%.")
    print("  looping needs a repetition penalty; long needs a bigger max_tokens. "
          "Fixing one leaves the other.")
    if partial:
        print(f"\n  ⚠ PARTIAL: {', '.join(sorted(partial))} — the think states have UNEQUAL n, so at "
              "least one\n    cell is still being written and its rate will move. Re-read when "
              "`qstat -u $USER` is empty.")


    if verdicts:
        print("\n  GATE (configs/wmdp7.yaml): the cell is usable when the model TERMINATES NORMALLY "
              "—\n  p50 far from the cap — and the residual truncation is a tail, not the median.")
        for (m, th), was, now, at_cap in verdicts:
            if at_cap:
                v = "FAIL — p50 is still the cap; the model does not terminate"
            elif now <= 0.10:
                v = "PASS"
            elif now < was / 2:
                v = (f"PARTIAL — {was:.1%} -> {now:.1%} is more than halved and p50 is off the cap, "
                     "so\n           the median generation is healthy and the rest is a tail. "
                     "Usable if\n           the parse rate clears ~90%; check score_accuracy before "
                     "sweeping.")
            else:
                v = f"FAIL — {was:.1%} -> {now:.1%} is not enough of a fall"
            print(f"    {m} think-{th}: {v}")

    for model, pid, item, sc, tail in samples:
        print(f"\n--- {model} · {pid} · {item}  (loop score {sc:.2f}) ---\n{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
