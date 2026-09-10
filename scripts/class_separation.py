#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dependency import compute_rows, mmlu_subject_questions  # noqa: E402
from src.paths import items_path  # noqa: E402


def concepts_from_items(items_path: Path) -> list[tuple[str, str]]:
    seen: dict[tuple[str, str], None] = {}
    with items_path.open() as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                seen[(it["concept_class"], it["concept_id"])] = None

    return sorted(seen, key=lambda k: (k[0] != "general", k[1]))


def write_figure(rows: list[dict], out: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        return f"(figure skipped: {e})"

    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, color in (("isolated", "tab:blue"), ("general", "tab:red")):
        pts = [(r["wordnet_subtree"] + 1, r["cross_subject_frac"]) for r in rows
               if r["concept_class"] == cls]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=color, label=cls, alpha=0.7, s=60)
    for r in rows:
        if r["concept_class"] == "general":
            ax.annotate(r["concept_id"], (r["wordnet_subtree"] + 1, r["cross_subject_frac"]))
    ax.set_xscale("log")
    ax.set_xlabel("WordNet hyponym-subtree size + 1 (log)")
    ax.set_ylabel("cross-MMLU-subject mention fraction")
    ax.set_title("Class separation: dependency proxies (isolated vs. general)")
    ax.legend()
    fig.tight_layout()
    path = out / "figure.png"
    fig.savefig(path, dpi=150)
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--items", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.run and not (args.items and args.out):
        raise SystemExit("pass --run exp1|round1 (or --items and --out explicitly)")
    items_file = Path(args.items) if args.items else items_path(args.run)
    out = Path(args.out) if args.out else Path("results") / args.run / "class_separation"

    concepts = concepts_from_items(items_file)
    print(f"{len(concepts)} concepts; loading MMLU 'all' for the cross-subject proxy…")
    rows = compute_rows(concepts, mmlu_subject_questions())

    out.mkdir(parents=True, exist_ok=True)
    with (out / "table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


    print(f"\n{'class':9} {'concept':22} {'wn_subtree':>10} {'cross_subj':>11}")
    gen = [r for r in rows if r["concept_class"] == "general"]
    iso = [r for r in rows if r["concept_class"] == "isolated"]
    for r in gen + iso[:5]:
        print(f"{r['concept_class']:9} {r['concept_id'][:22]:22} "
              f"{r['wordnet_subtree']:>10} {r['cross_subject_frac']:>11}")
    if len(iso) > 5:
        print(f"{'':9} …and {len(iso) - 5} more isolated entities")

    def avg(key, cls):
        vals = [r[key] for r in rows if r["concept_class"] == cls]
        return sum(vals) / len(vals) if vals else 0.0

    print("\n-- separation (mean) --")
    for key in ("wordnet_subtree", "cross_subject_frac"):
        g, i = avg(key, "general"), avg(key, "isolated")
        ok = "OK isolated<general" if g > i else "!! NOT ordered — inspect"
        print(f"  {key:20} general={g:.3f}  isolated={i:.3f}   {ok}")

    print(f"\ntable -> {out/'table.csv'}\nfigure -> {write_figure(rows, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
