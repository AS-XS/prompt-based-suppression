#!/usr/bin/env python


from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.sample_traces import lexical_flags  # noqa: E402  (the proxy flags of the first sample)
from src.analysis.panel import (BASELINE, Cell, appended_tail, arm_of, draw,  # noqa: E402
                                file_order, method_of, same_instrument, target_concept)
from src.analysis.surfaces import trace_of  # noqa: E402
from src.paths import generations_glob, judge_samples_dir  # noqa: E402
from src.provenance import git_commit  # noqa: E402

UNLEARN_RUNS = ("rmu1", "rmu2", "rmu3", "rmu4", "elm1", "elm2", "elm3", "r2mu-e2b", "r2mu-g31")


CALLS_PER_SURFACE, TOKENS_PER_CALL, TOKENS_PER_SECOND = 3, 1480, 937


def _gid(r: dict) -> str:
    return f"{r['config_hash']}:{r['item_id']}:{r['condition']}:{r['think_state']}"


def scan(run: str, prompt_run: str, models, think, probe, cells, base_cfg, sources, seen,
         skip=frozenset()) -> int:


    dupes = 0
    paths = sorted(Path().glob(generations_glob(run)))
    if run == prompt_run and models:


        paths = [p for p in paths if p.name.split("__")[0] in models
                 or p.name.split("__")[1].startswith(BASELINE)]
    for p in paths:
        digest, offset = hashlib.sha256(), 0
        with open(p, "rb") as f:
            for line in f:
                digest.update(line)
                start, offset = offset, offset + len(line)
                if not line.strip():
                    continue
                r = json.loads(line)
                model, ckpt = r.get("model"), method_of(r.get("model") or "") is not None
                if r.get("prompt_id") == BASELINE and not ckpt:
                    base_cfg[run][model].add(r["config_hash"])
                if r.get("probe_type") != probe or r.get("think_state") not in think:
                    continue
                if run == prompt_run:
                    if ckpt or (models and model not in models) or model in skip:
                        continue
                elif not ckpt or r.get("concept_id") != target_concept(model):
                    continue
                if r["think_state"] == "on" and not (trace_of(r.get("raw_output")) or "").strip():
                    continue
                gid = _gid(r)
                if gid in seen:
                    dupes += 1
                    continue
                seen.add(gid)
                cell = Cell(run, model, r["concept_id"], r["prompt_id"], r["think_state"])
                held = cells[cell].get(r["item_id"])
                if held is not None:
                    raise SystemExit(f"{cell} holds item {r['item_id']} under two generation_ids "
                                     f"({held[3]} and {gid}): two configurations of one model. "
                                     f"Sample one configuration, not a mixture.")
                cells[cell][r["item_id"]] = (p, start, len(line), gid)
        sources[f"{run}/{p.name}"] = {"sha256": digest.hexdigest(), "bytes": offset}
    return dupes


def _existing_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [_gid(json.loads(line)) for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-run", default="wmdp8")
    ap.add_argument("--unlearn-runs", nargs="*", default=list(UNLEARN_RUNS))
    ap.add_argument("--home", default="wmdp8-judge", help='Home.')
    ap.add_argument("--per-concept", type=int, default=100,
                    help='Per concept.')
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--name", default=None, help='Name.')
    ap.add_argument("--models", nargs="+", default=None,
                    help='Models.')
    ap.add_argument("--skip-models", nargs="+", default=[],
                    help='Skip models.')
    ap.add_argument("--think", nargs="+", default=["on", "na"])
    ap.add_argument("--probe", default="forget")
    ap.add_argument("--force", action="store_true",
                    help='Force.')
    args = ap.parse_args(argv)

    name = args.name or f"panel-s{args.seed}"
    out_dir = judge_samples_dir(args.home)
    cells: dict[Cell, dict] = defaultdict(dict)
    base_cfg: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    sources, seen, dupes = {}, set(), 0
    for run in [args.prompt_run, *args.unlearn_runs]:
        if not any(Path().glob(generations_glob(run))):
            print(f"  {run}: no generations here — skipped", flush=True)
            continue
        print(f"scanning {run} ...", flush=True)
        dupes += scan(run, args.prompt_run, set(args.models or ()), set(args.think), args.probe,
                      cells, base_cfg, sources, seen, frozenset(args.skip_models))
    taken = {}
    for run in args.unlearn_runs:
        if run in base_cfg:
            ok, why = same_instrument(base_cfg[run], base_cfg[args.prompt_run])
            taken[run] = {"included": ok, "why": why}
            print(f"  {run}: {'taken' if ok else 'LEFT OUT'} — {why}", flush=True)
            if not ok:
                for cell in [c for c in cells if c.run == run]:
                    del cells[cell]
    if not cells:
        raise SystemExit("no eligible rows")

    picks, short = draw({c: set(items) for c, items in cells.items()}, args.per_concept, args.seed)
    for cell, want, got in short:
        print(f"  ⚠️  {'|'.join(cell)}: asked {want}, only {got} eligible — took all", flush=True)
    order = file_order(picks, args.prompt_run)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{name}.manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    touched = {f"{name}__{part}.jsonl" for part in order}

    rows_meta = {g: m for g, m in old.get("rows", {}).items() if m["file"] not in touched}
    files = {f: v for f, v in old.get("files", {}).items() if f not in touched}
    new_rows = []
    for part, rows in order.items():
        path = out_dir / f"{name}__{part}.jsonl"
        drawn = [cells[cell][item][3] for _, cell, item in rows]
        existing = _existing_ids(path)
        try:
            tail = appended_tail(existing, drawn)
        except ValueError as e:
            if not args.force:
                raise SystemExit(f"{path.name}: {e}\nNothing was written to it. --force rewrites "
                                 f"it, and every judgment of it then has to be redone.")
            existing, tail = [], drawn
        by_file = defaultdict(list)
        for rank, cell, item in rows:
            by_file[cells[cell][item][0]].append((rank, cell, item))
        raw = {}
        for src, addrs in by_file.items():
            with open(src, "rb") as f:
                for rank, cell, item in addrs:
                    _, start, length, gid = cells[cell][item]
                    f.seek(start)
                    line = f.read(length)
                    raw[gid] = line if line.endswith(b"\n") else line + b"\n"
                    r = json.loads(line)
                    base = (cell.model if cell.run == args.prompt_run
                            else next(iter(base_cfg[cell.run])))
                    rows_meta[gid] = {
                        "run": cell.run, "model": cell.model, "method": method_of(cell.model),
                        "base_model": base, "arm": arm_of(cell), "concept_id": cell.concept_id,
                        "condition": r.get("condition"), "prompt_id": cell.prompt_id,
                        "think_state": cell.think_state, "item_id": item, "rank": rank,
                        "file": path.name, "source": f"{cell.run}/{src.name}",
                        "finish_reason": r.get("finish_reason"), **lexical_flags(r)}
        with open(path, "ab" if existing else "wb") as f:
            for gid in tail:
                f.write(raw[gid])
        new_rows += [rows_meta[g] for g in tail]
        files[path.name] = {"rows": len(drawn), "appended": len(tail),
                            "per_concept": args.per_concept}

    manifest = {
        "design": "panel", "name": name, "home": args.home, "prompt_run": args.prompt_run,
        "probe": args.probe, "seed": args.seed, "per_concept": args.per_concept,
        "history": old.get("history", []) + [{"per_concept": args.per_concept, "models": args.models,
                                               "at": datetime.now(timezone.utc).isoformat()}],
        "think": args.think, "unlearn_runs": {**old.get("unlearn_runs", {}), **taken},
        "created_at": old.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "population": {**old.get("population", {}),
                       **{"|".join(c): len(v) for c, v in sorted(cells.items())}},
        "allocation": {**old.get("allocation", {}),
                       **{"|".join(c): len(v) for c, v in sorted(picks.items())}},
        "shortfalls": [{"cell": "|".join(c), "asked": w, "eligible": g} for c, w, g in short],
        "duplicates_skipped": dupes, "files": files, "rows": rows_meta,
        "sources": {**old.get("sources", {}), **sources}}
    manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    def surfaces(ms) -> int:
        return sum((m["think_state"] == "on") + bool(m["has_explanation"]) for m in ms)

    def gpu_hours(ms) -> float:
        return surfaces(ms) * CALLS_PER_SURFACE * TOKENS_PER_CALL / TOKENS_PER_SECOND / 3600

    print(f"\n{'file':<58} {'rows':>7} {'new':>7} {'surfaces':>9} {'GPU-h (new)':>12}")
    for fname in sorted(touched):
        mine = [m for m in rows_meta.values() if m["file"] == fname]
        new = [m for m in new_rows if m["file"] == fname]
        print(f"{fname:<58} {len(mine):>7,} {len(new):>7,} {surfaces(mine):>9,} "
              f"{gpu_hours(new):>12.1f}")
    print(f"\n~{gpu_hours(new_rows):.0f} GPU-hours to judge the {len(new_rows):,} new rows (estimate: "
          f"{CALLS_PER_SURFACE} calls a surface, {TOKENS_PER_CALL} tokens a call, "
          f"{TOKENS_PER_SECOND} tokens/s a job)\nmanifest -> {manifest_path}")
    if dupes:
        print(f"({dupes} duplicate generation_id row(s) skipped: the FIRST copy is kept, the rule "
              f"score_accuracy follows — e.g. Qwen3-14B's baseline shards 2-4, findings 2026-10-09)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
