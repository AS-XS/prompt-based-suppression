#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re  # noqa: E402

import yaml  # noqa: E402

from scripts.generate import load_items  # noqa: E402  -- the SAME selection generation performs
from src.gen.thinkstate import think_states_for  # noqa: E402
from src.paths import generations_dir, items_path  # noqa: E402


_SHARD_SUFFIX = re.compile(r"\.shard\d+of\d+$")


def selected_items(items_file: Path, cfg: dict) -> int:


    return len(load_items(items_file, cfg.get("concepts"), cfg.get("probe_types"),
                          cfg.get("max_items")))


def _gen_id(d: dict) -> str:

    return f"{d.get('config_hash')}:{d.get('item_id')}:{d.get('condition')}:{d.get('think_state')}"


def _scan(path: Path) -> tuple[int, Counter, Counter, Counter]:


    rows, states, finish, ids = 0, Counter(), Counter(), Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                finish["UNPARSEABLE LINE"] += 1     
                continue
            rows += 1
            states[d.get("think_state")] += 1
            finish[d.get("finish_reason")] += 1
            ids[_gen_id(d)] += 1
    return rows, states, finish, ids


_FEWSHOT_SLACK = 200


def _is_fewshot(wave: str) -> bool:
    return "fewshot" in wave


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp3")
    ap.add_argument("--models", nargs="+", default=None,
                    help='Models.')
    ap.add_argument("--waves", nargs="+", default=None, help='Waves.')
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--config", default=None,
                    help='Config.')
    args = ap.parse_args()

    items_file = Path(args.items) if args.items else items_path(args.run)
    cfg_path = Path(args.config) if args.config else Path(f"configs/{args.run}.yaml")
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    n_items = selected_items(items_file, cfg) if items_file.exists() else 0
    if cfg:
        print(f"config {cfg_path} · probe_types={cfg.get('probe_types')}")
    else:
        print(f"⚠ no config at {cfg_path} — expected counts assume NO probe_type filter, which "
              f"over-states them if the item set holds `dependent` probes")
    gdir = generations_dir(args.run)
    files = sorted(glob.glob(str(gdir / "*.jsonl")))
    if not files:
        print(f"no generations under {gdir}")
        return 1


    seen: dict[str, dict[str, list[Path]]] = {}
    for p in files:
        model, _, wave = Path(p).stem.partition("__")
        wave = _SHARD_SUFFIX.sub("", wave)
        seen.setdefault(model, {}).setdefault(wave, []).append(Path(p))

    models = sorted(args.models or seen)


    stated = args.waves is not None
    waves = sorted(args.waves or {w for m in seen.values() for w in m})
    if not n_items:


        best = 0
        for m, ws in seen.items():
            for w, paths in ws.items():
                per_state: dict = {}
                for p in paths:
                    with p.open(encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                d = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            per_state.setdefault(d.get("think_state"), set()).add(_gen_id(d))
                best = max(best, max((len(v) for v in per_state.values()), default=0))
        n_items = best
        print(f"(no items.jsonl for run {args.run!r} — assuming {n_items:,} items, "
              f"the largest per-think-state count seen)\n")

    if not args.models:
        print("(models discovered from files present — one that wrote NOTHING cannot appear here; "
              "pass --models to check a full roster)")
    print(f"run {args.run} · {n_items:,} items selected · expected = items x think-states\n")
    print(f"{'model':28} {'states':8} {'wave':24} {'rows':>9} {'expected':>9}  status")
    print("-" * 96)
    absent: list = []
    problems: list[tuple[str, str, str]] = []
    for m in models:
        try:
            states = think_states_for(m)
        except (KeyError, SystemExit, ValueError) as e:


            print(f"{m:28} {'?':8} {'—':24} {'—':>9} {'—':>9}  NOT IN think_states.yaml")
            print(f"{'':28} {str(e).splitlines()[0][:80]}")
            problems.append((m, "*", "unaudited model"))
            continue
        expected_full = n_items * len(states)
        for w in waves:


            expected = expected_full
            paths = seen.get(m, {}).get(w)
            if not paths:
                label = "MISSING" if stated else "— not submitted"
                print(f"{m:28} {str(states):8} {w:24} {'-':>9} {expected:>9,}  {label}")
                (problems if stated else absent).append((m, w, "missing"))
                continue
            rows, per_state, finish, ids = 0, Counter(), Counter(), Counter()
            per_file = {}
            for p in paths:
                r, st, fi, idc = _scan(p)
                rows += r
                per_state.update(st)
                finish.update(fi)
                ids.update(idc)
                per_file[p] = set(idc)
            if _is_fewshot(w):

                short = expected_full - rows
                if 0 < short <= _FEWSHOT_SLACK * len(states):
                    print(f"{m:28} {str(states):8} {w + shards:24} {rows:>9,} "
                          f"{expected_full:>9,}  complete (−{short:,}: {short // max(len(states),1)} "
                          f"held-out example item(s) × {len(states)} think-state(s))")
                    continue
            if rows == expected:
                status = "ok"
            elif rows < expected:
                status = f"SHORT by {expected - rows:,} ({100*rows/expected:.0f}% done)"
                problems.append((m, w, status))
            else:
                dupes = sum(c - 1 for c in ids.values() if c > 1)
                status = (f"OVER by {rows - expected:,} — {dupes:,} DUPLICATE cells"
                          if dupes else f"OVER by {rows - expected:,} (not duplicates: more rows "
                                        f"than items?)")
                problems.append((m, w, status))
            shards = f" [{len(paths)} shards]" if len(paths) > 1 else ""
            print(f"{m:28} {str(states):8} {w + shards:24} {rows:>9,} {expected:>9,}  {status}")

            if rows != expected and per_state:
                detail = " ".join(f"{k}={v:,}" for k, v in sorted(per_state.items(), key=str))
                print(f"{'':28} {'':8} {'':24} {'':>9} {'':>9}    by state: {detail}")


            if rows > expected and len(paths) > 1:
                shared = [(a.name, b.name, len(per_file[a] & per_file[b]))
                          for i, a in enumerate(paths) for b in paths[i + 1:]
                          if per_file[a] & per_file[b]]
                for an, bn, n in sorted(shared, key=lambda x: -x[2])[:4]:
                    print(f"{'':28} {'':8} {'':24} {'':>9} {'':>9}    overlap: {an} ∩ {bn} "
                          f"= {n:,} cells")
                if shared:
                    print(f"{'':28} {'':8} {'':24} {'':>9} {'':>9}    -> delete the file that is NOT "
                          f"a shard; the shards cover the whole wave")

            trunc = finish.get("length", 0)
            if rows and trunc / rows > 0.10:
                print(f"{'':28} {'':8} {'':24} {'':>9} {'':>9}    ⚠ {100*trunc/rows:.0f}% hit the "
                      f"token cap (finish_reason=length)")

    print()
    if absent:
        by_model = Counter(m for m, _, _ in absent)
        print(f"{len(absent)} cell(s) NOT SUBMITTED for that model "
              f"({', '.join(f'{m} {n}' for m, n in sorted(by_model.items()))}).")
        print("  Not necessarily a problem: a run may give different models different waves on")
        print("  purpose. But if you DID submit one of these, it wrote ZERO rows — the silent-zero")
        print("  failure this tool exists to catch. Pass --waves to state the intended set and")
        print("  have these counted as real gaps.")
        print()
    if not problems:
        print(f"complete: {len(models)} model(s) x {len(waves)} wave(s) — every cell that exists is full.")
        return 0
    print(f"{len(problems)} incomplete cell(s). Re-run the identical generate line — the sweep is")
    print("resumable per generation_id, so it fills only what is missing:")
    runnable = [(m, w) for m, w, _ in problems if w != "*"]     
    for m, w in runnable[:8]:
        cond = "baseline" if w == "none-baseline" else "suppressed"
        extra = "" if w == "none-baseline" else f" --prompt-suppressed {w}"
        print(f"  python scripts/generate.py --config configs/{args.run}.yaml "
              f"--model {m} --conditions {cond}{extra}")
    if len(runnable) > 8:
        print(f"  ... and {len(runnable) - 8} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
