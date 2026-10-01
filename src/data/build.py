from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.mmlu import CONCEPT_SUBJECTS, load_mmlu_concept
from src.data.probes import load_dependent
from src.data.retain import load_retain, load_utility
from src.data.rwku import load_rwku
from src.data.wmdp import SUBSETS as WMDP_SUBSETS, load_near_retain, load_wmdp
from src.data.schema import Item
from src.paths import items_path


GENERAL_CONCEPTS = list(CONCEPT_SUBJECTS)


def build_wmdp_items(subsets: list[str], near_limit: int | None = 100,
                     utility_per_subject: int = 15,
                     forget_limit: int | None = None) -> list[Item]:


    items: list[Item] = []
    for sub in subsets:
        cid = sub if sub.startswith("wmdp-") else f"wmdp-{sub}"
        if cid not in WMDP_SUBSETS:
            raise SystemExit(f"unknown WMDP subset {sub!r}; expected bio | cyber | chem")
        forget = load_wmdp(cid)
        items += forget[:forget_limit] if forget_limit else forget
        items += load_near_retain(cid, limit_per_subject=near_limit)
        items += load_utility(cid, per_subject=utility_per_subject)
    items, dropped = dedupe(items)
    if dropped:
        print(f"deduped {dropped} repeated item_id(s)")
    return items


def build_items(entities_n: int = 200, mmlu_limit: int | None = 120,
                concepts: list[str] | None = None, retain_limit: int | None = 50,
                open_ended: bool = False) -> list[Item]:
    concepts = concepts or GENERAL_CONCEPTS
    items: list[Item] = []


    items += load_rwku(n=entities_n)


    for c in concepts:
        items += load_mmlu_concept(c, open_ended=False, limit_per_subject=mmlu_limit)
        if open_ended:
            items += load_mmlu_concept(c, open_ended=True, limit_per_subject=mmlu_limit)
        items += load_retain(c, limit_per_subject=retain_limit)
        items += load_utility(c)
        items += load_dependent(c)
    items, dropped = dedupe(items)
    if dropped:
        print(f"deduped {dropped} repeated item_id(s)")
    return items


def dedupe(items: list[Item]) -> tuple[list[Item], int]:


    seen: dict[str, Item] = {}
    for it in items:
        seen.setdefault(it.item_id, it)
    return list(seen.values()), len(items) - len(seen)


def stats_table(items: list[Item]) -> dict:
    c: Counter = Counter((it.concept_class, it.concept_id, it.probe_type) for it in items)
    return {f"{cls}/{cid}/{pt}": n for (cls, cid, pt), n in sorted(c.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--out", type=str, default=None, help='Out.')
    ap.add_argument("--entities", type=int, default=200,
                    help='Entities.')
    ap.add_argument("--mmlu-limit", type=int, default=120, help='Mmlu limit.')
    ap.add_argument("--retain-limit", type=int, default=50,
                    help='Retain limit.')
    ap.add_argument("--wmdp", nargs="+", default=None, metavar="SUBSET",
                    help='Wmdp.')
    ap.add_argument("--near-limit", type=int, default=100,
                    help='Near limit.')
    ap.add_argument("--wmdp-limit", type=int, default=None, metavar="N",
                    help='Wmdp limit.')
    ap.add_argument("--rwku", action="store_true",
                    help='Rwku.')
    ap.add_argument("--open-ended", action="store_true",
                    help='Open ended.')
    args = ap.parse_args()

    if not args.out and not args.run:
        raise SystemExit("pass --run exp1|round1 (or an explicit --out): the studies must not share "
                         "an item set")
    if args.wmdp and args.rwku:
        raise SystemExit("--wmdp and --rwku each build a whole item set; pass one")
    if args.rwku:
        items, dropped = dedupe(load_rwku(n=args.entities))
        if dropped:
            print(f"deduped {dropped} repeated item_id(s)")
    elif args.wmdp:
        items = build_wmdp_items(args.wmdp, near_limit=args.near_limit,
                                 forget_limit=args.wmdp_limit)
    else:
        items = build_items(entities_n=args.entities, mmlu_limit=args.mmlu_limit,
                            retain_limit=args.retain_limit, open_ended=args.open_ended)

    out = Path(args.out) if args.out else items_path(args.run).parent
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")

    table = stats_table(items)
    (out / "stats.json").write_text(json.dumps(table, indent=2))
    print(f"wrote {len(items)} items -> {out/'items.jsonl'}")
    for k, v in table.items():
        print(f"  {k:45s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
