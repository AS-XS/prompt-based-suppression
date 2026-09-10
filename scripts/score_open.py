#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.surfaces import answer_of  # noqa: E402
from src.data.open_quality import gold_answer, is_self_contained, open_ended_correct  # noqa: E402
from src.gen.runner import iter_jsonl  # noqa: E402
from src.paths import analysis_dir, generations_glob, items_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp1")
    ap.add_argument("--threshold", type=float, default=0.6, help='Threshold.')
    args = ap.parse_args()

    ip = items_path(args.run)
    if not ip.exists():
        raise SystemExit(f"need items.jsonl for the questions + gold answers, not found: {ip}\n"
                         "  run this on the node that has data/processed/<run>/items.jsonl")


    items: dict[str, dict] = {}
    for it in iter_jsonl(ip):
        if (it.get("meta") or {}).get("format") != "open_ended":
            continue
        q = it.get("question")
        items[it["item_id"]] = {"gold": gold_answer(it), "ok": is_self_contained(q)}

    n_open = len(items)
    n_ok = sum(v["ok"] for v in items.values())
    print(f"open-ended items: {n_open} · self-contained (kept): {n_ok} · dropped (need options): "
          f"{n_open - n_ok}  ({(n_open-n_ok)/n_open:.0%})")


    cells: dict[tuple, list[bool]] = defaultdict(list)
    for p in sorted(glob.glob(generations_glob(args.run))):
        for g in iter_jsonl(p):
            it = items.get(g.get("item_id"))
            if not it or not it["ok"] or not it["gold"]:
                continue
            key = (g.get("model"), g.get("think_state"), g.get("prompt_id"))
            cells[key].append(open_ended_correct(answer_of(g.get("raw_output")), it["gold"],
                                                 threshold=args.threshold))

    rows = []
    for (model, ts, prompt), hits in sorted(cells.items(), key=lambda kv: str(kv[0])):
        n = len(hits)
        rows.append({"model": model, "think_state": ts, "prompt_id": prompt,
                     "n": n, "open_acc": round(sum(hits) / n, 4) if n else ""})

    out = analysis_dir(args.run) / "open_accuracy.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "think_state", "prompt_id", "n", "open_acc"])
        w.writeheader()
        w.writerows(rows)
    print(f"open-ended accuracy -> {out}  ({len(rows)} cells; threshold={args.threshold})")
    print("  compare open_acc (no floor) against MC accuracy (25% floor): the gap is how much the "
          "chance floor hides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
