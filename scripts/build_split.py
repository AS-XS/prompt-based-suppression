#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import Item  # noqa: E402
from src.gen.runner import read_jsonl  # noqa: E402
from src.paths import items_path, split_path  # noqa: E402


def build(items: list[Item], val_frac: float, seed: int) -> dict[str, list[str]]:


    by_concept: dict[str, list[str]] = {}
    for it in items:
        if it.probe_type == "forget":
            by_concept.setdefault(it.concept_id, []).append(it.item_id)
    out = {}
    for concept, ids in sorted(by_concept.items()):
        ids = sorted(ids)
        rng = random.Random(seed)
        rng.shuffle(ids)
        n_val = round(len(ids) * val_frac)
        out[concept] = sorted(ids[:n_val])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="wmdp1")
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--out", default=None, help='Out.')
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help='Check.')
    ap.add_argument("--force", action="store_true",
                    help='Force.')
    args = ap.parse_args()

    items_file = Path(args.items) if args.items else items_path(args.run)
    out_file = Path(args.out) if args.out else split_path(args.run)
    if not items_file.exists():
        raise SystemExit(f"no item set at {items_file} — build it first:\n"
                         f"  python -m src.data.build --run {args.run} ...")
    items = list(read_jsonl(items_file, Item))
    forget = [it for it in items if it.probe_type == "forget"]
    if not forget:
        raise SystemExit(f"no forget items in {items_file} — nothing to split")

    if args.check or (out_file.exists() and not args.force):
        if not out_file.exists():
            raise SystemExit(f"no split at {out_file} — build it: {sys.argv[0]} --run {args.run}")
        split = json.loads(out_file.read_text(encoding="utf-8"))
        val = {c: set(v) for c, v in split["val"].items()}
        print(f"split {out_file}  (seed {split.get('seed')}, val_frac {split.get('val_frac')})")
        print(f"{'concept':16} {'val':>7} {'test':>7} {'total':>7}  status")
        print("-" * 56)
        ok = True
        have = Counter(it.concept_id for it in forget)
        ids_by_concept: dict[str, set] = {}
        for it in forget:
            ids_by_concept.setdefault(it.concept_id, set()).add(it.item_id)
        for concept in sorted(have):
            v = val.get(concept, set())
            all_ids = ids_by_concept[concept]
            missing = v - all_ids
            status = "ok"
            if missing:
                status = f"STALE — {len(missing):,} val id(s) absent from items.jsonl"
                ok = False
            print(f"{concept:16} {len(v):>7,} {len(all_ids) - len(v):>7,} {len(all_ids):>7,}  {status}")
        for concept in sorted(set(val) - set(have)):
            print(f"{concept:16} {'—':>7} {'—':>7} {'—':>7}  STALE — concept not in items.jsonl")
            ok = False
        if not ok:
            print("\nThe split does not match the item set. Rebuilding the items with a different")
            print("limit or source changes the id space; a split from before that is not valid.")
            return 1
        print("\nsplit matches the item set.")
        if not args.check and out_file.exists():
            print("Existing split left alone — pass --force to overwrite (see --force help).")
        return 0

    val = build(items, args.val_frac, args.seed)
    payload = {
        "note": "WMDP ships only a `test` split; this val/test division is OURS and must be cited. "
                "Tune on val, report on test. Rebuilding with a different seed invalidates every "
                "comparison already made.",
        "seed": args.seed,
        "val_frac": args.val_frac,
        "val": val,
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {out_file}  (seed {args.seed}, val_frac {args.val_frac})")
    print(f"\n{'concept':16} {'val':>7} {'test':>7} {'total':>7}")
    print("-" * 42)
    have = Counter(it.concept_id for it in forget)
    for concept in sorted(have):
        n, v = have[concept], len(val.get(concept, []))
        print(f"{concept:16} {v:>7,} {n - v:>7,} {n:>7,}")
    print("\nCOMMIT this file — it is part of the method, not scratch state.")
    print("After the baseline run, confirm val and test have similar baseline accuracy; if they do")
    print("not, reseed NOW, before any tuning has looked at either half.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
