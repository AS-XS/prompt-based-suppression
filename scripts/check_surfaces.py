#!/usr/bin/env python


from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.surfaces import TAG_PAIRS, split_think  # noqa: E402


MARKER_RATE = 0.05


_TAGISH = re.compile(r"<\|?/?[A-Za-z_][A-Za-z0-9_.\-]*\|?>")
_KNOWN = {t for pair in TAG_PAIRS for t in pair}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help='Inputs.')
    ap.add_argument("--examples", type=int, default=2, help='Examples.')
    args = ap.parse_args()

    paths = [f for pat in args.inputs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched: {args.inputs}")

    cells: dict[tuple, dict] = collections.defaultdict(
        lambda: {"n": 0, "trace": 0, "empty": 0, "markers": collections.Counter(), "eg": []})
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r.get("model"), r.get("think_state"))
            c = cells[k]
            c["n"] += 1
            trace, answer = split_think(r.get("raw_output"))
            if trace is not None:
                c["trace"] += 1
            if not answer.strip():
                c["empty"] += 1
            for m in _TAGISH.findall(answer or ""):
                if m not in _KNOWN:
                    c["markers"][m] += 1
                    if len(c["eg"]) < args.examples:
                        c["eg"].append((answer or "")[:200].replace("\n", " "))

    print(f"TAG_PAIRS knows: {[list(p) for p in TAG_PAIRS]}\n")
    print(f"{'model':24} {'think':6} {'rows':>7} {'trace found':>12} {'empty answer':>13}  unknown tags in ANSWER")
    bad = False
    for (model, ts), c in sorted(cells.items(), key=lambda kv: str(kv[0])):
        frac = c["trace"] / c["n"] if c["n"] else 0
        top = ", ".join(f"{t}×{n}" for t, n in c["markers"].most_common(3)) or "-"
        print(f"{str(model):24} {str(ts):6} {c['n']:7} {frac:11.1%} {c['empty']/c['n']:12.1%}  {top}")

        if ts == "on" and frac < 0.5:
            bad = True
            print(f"    🚩 think-on but a trace was found in only {frac:.1%} of rows — TAG_PAIRS is\n"
                  f"       probably missing this model's opening delimiter, so the REASONING IS BEING\n"
                  f"       SCORED AS THE ANSWER (silent). Read the pair off its chat template.")


        marker_rows = sum(c["markers"].values())
        rate = marker_rows / c["n"] if c["n"] else 0
        if c["markers"] and rate >= MARKER_RATE:
            bad = True
            print(f"    🚩 tag-shaped tokens in {rate:.1%} of answers — likely an unknown pair.")
            for e in c["eg"]:
                print(f"       e.g. {e}")
        elif c["markers"]:
            print(f"    ·  tag-shaped tokens on {marker_rows} of {c['n']} rows ({rate:.2%}) — below "
                  f"the {MARKER_RATE:.0%} threshold, so read as CONTENT, not a delimiter.")
            print(f"       WMDP-cyber carries markup and shell syntax in the questions themselves. "
                  f"Do NOT add these to TAG_PAIRS; check one example before acting.")
            for e in c["eg"][:1]:
                print(f"       e.g. {e}")

    print()
    print("OK — every think-on cell split a trace, no unknown tags in answers." if not bad else
          "ACTION: teach src/analysis/surfaces.py::TAG_PAIRS the missing pair, then re-score.\n"
          "Nothing needs regenerating — surfaces.py is a re-runnable pass over stored raw_output.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
