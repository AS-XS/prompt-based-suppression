#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts, write_csv as _write_csv  # noqa: E402
from src.analysis.recall import recall_delta, recall_table, score_recall  # noqa: E402
from src.paths import analysis_dir, generations_glob, items_path  # noqa: E402
from src.provenance import select_one_config  # noqa: E402


def _item_index(path: Path, subtype: str | None) -> dict[str, dict]:
    idx = {}
    for it in _read_dicts(path):
        if subtype and it.get("concept_subtype") != subtype:
            continue
        idx[it["item_id"]] = {"reference": it.get("reference"), "probe_type": it.get("probe_type"),
                              "concept_id": it.get("concept_id")}
    return idx


def _delta_cols(base: list[str], supp: list[str]) -> list[str]:
    return base + ["baseline", "n_baseline"] + [c for p in supp
                                                 for c in (p, f"{p}_drop", f"{p}_n")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--generations", nargs="+", default=None, help='Generations.')
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--config-hash", default=None,
                    help='Config hash.')
    ap.add_argument("--threshold", type=float, default=0.6, help='Threshold.')
    ap.add_argument("--subtype", default="entity", help='Subtype.')
    args = ap.parse_args()

    if not args.run and not (args.items and args.generations):
        raise SystemExit("pass --run exp1|round1 (or --items and --generations explicitly)")
    items_file = Path(args.items) if args.items else items_path(args.run)
    gen_globs = args.generations or [generations_glob(args.run)]
    out_dir = Path(args.out_dir) if args.out_dir else analysis_dir(args.run)

    idx = _item_index(items_file, args.subtype or None)
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no generation files matched: {gen_globs}")
    gens = [g for p in paths for g in _read_dicts(Path(p))]
    gens = select_one_config(gens, args.config_hash, what="generations")

    scored = score_recall(gens, idx, threshold=args.threshold)
    if not scored:
        raise SystemExit(f"no recall-scorable generations (subtype={args.subtype!r} items with a "
                         "reference, and generations over them?)")

    out = out_dir


    table = recall_table(scored)
    _write_csv(out / "recall.csv", table, list(table[0]) if table else [])

    forget = [s for s in scored if s["probe_type"] == "forget"]
    d_f, sp_f = recall_delta(forget, by=("model", "think_state", "concept_id"))
    _write_csv(out / "recall_delta_forget.csv", d_f,
               _delta_cols(["model", "think_state", "concept_id"], sp_f))

    neighbor = [s for s in scored if s["probe_type"] == "neighbor"]
    d_n, sp_n = recall_delta(neighbor, by=("model", "think_state", "concept_id"))
    _write_csv(out / "recall_delta_neighbor.csv", d_n,
               _delta_cols(["model", "think_state", "concept_id"], sp_n))

    print(f"scored {len(scored)} recall generations over {len(paths)} file(s)")
    print(f"  recall            -> {out / 'recall.csv'}")
    print(f"  on-target (forget)-> {out / 'recall_delta_forget.csv'}  (prompts {sp_f})")
    print(f"  collateral (neigh)-> {out / 'recall_delta_neighbor.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
