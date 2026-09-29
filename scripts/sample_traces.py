#!/usr/bin/env python


from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.surface_gap import leak_pattern, leaks, novel_pattern  # noqa: E402
from src.analysis.surfaces import explanation_of, trace_of  # noqa: E402
from src.paths import generations_glob, judge_samples_dir  # noqa: E402
from src.provenance import git_commit  # noqa: E402

BASELINE = "none-baseline"


def _gid(r: dict) -> str:

    return f"{r['config_hash']}:{r['item_id']}:{r['condition']}:{r['think_state']}"


def scan(paths: list[Path], probe: str, models=None):


    cells: dict[tuple, dict] = defaultdict(dict)
    sources, seen, duplicates = {}, set(), 0
    for p in paths:
        digest, offset = hashlib.sha256(), 0
        with open(p, "rb") as f:
            for line in f:
                digest.update(line)
                start, offset = offset, offset + len(line)
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get("think_state") != "on" or r.get("probe_type") != probe
                        or (models and r.get("model") not in models)):
                    continue
                if not (trace_of(r.get("raw_output")) or "").strip():
                    continue
                gid = _gid(r)
                if gid in seen:
                    duplicates += 1
                    continue
                seen.add(gid)
                key = (r["model"], r["concept_id"], r["condition"], r["prompt_id"])
                held = cells[key].get(r["item_id"])
                if held is not None:
                    raise SystemExit(
                        f"{key} holds item {r['item_id']} under two generation_ids ({held[3]} and "
                        f"{gid}): two configurations of one model in this run. Sample one run, "
                        f"not a mixture.")
                cells[key][r["item_id"]] = (p, start, len(line), gid)
        sources[p.name] = {"sha256": digest.hexdigest(), "bytes": offset}
    return cells, sources, duplicates


def draw(cells: dict, per_wave: int, n_baseline: int, seed: int):


    rng = random.Random(seed)
    picks, short = [], []
    for model, concept in sorted({(k[0], k[1]) for k in cells if k[2] == "suppressed"}):
        drawn = []
        for key in sorted(k for k in cells if k[:3] == (model, concept, "suppressed")):
            pool = sorted(cells[key])
            if len(pool) < per_wave:
                short.append((key, per_wave, len(pool)))
            chosen = rng.sample(pool, min(per_wave, len(pool)))
            picks += [(key, i) for i in chosen]
            drawn += chosen
        bkey = (model, concept, "baseline", BASELINE)
        eligible = sorted(set(drawn) & set(cells.get(bkey, {})))
        if len(eligible) < n_baseline:
            short.append((bkey, n_baseline, len(eligible)))
        picks += [(bkey, i) for i in rng.sample(eligible, min(n_baseline, len(eligible)))]
    return picks, short


def lexical_flags(r: dict) -> dict:


    cid, cls, pid = r["concept_id"], r["concept_class"], r["prompt_id"]
    pat, npat = leak_pattern(cid, cls), novel_pattern(pid, cid, cls)
    tr, ex = trace_of(r["raw_output"]) or "", explanation_of(r["raw_output"]) or ""
    has_ex = bool(ex.strip())
    return {"lex_trace": leaks(tr, pat),
            "lex_trace_novel": leaks(tr, npat) if npat is not None else None,
            "lex_explanation": leaks(ex, pat) if has_ex else None,
            "lex_explanation_novel": leaks(ex, npat) if (has_ex and npat is not None) else None,
            "trace_chars": len(tr), "has_explanation": has_ex}


def _k(key: tuple) -> str:
    return "|".join(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--probe", default="forget",
                    choices=["forget", "neighbor", "retain", "utility", "dependent"])
    ap.add_argument("--per-wave", type=int, default=4,
                    help='Per wave.')
    ap.add_argument("--baseline", type=int, default=29,
                    help='Baseline.')
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--models", nargs="+", default=None, help='Models.')
    ap.add_argument("--name", default=None, help='Name.')
    ap.add_argument("--out-dir", default=None, help='Out dir.')
    ap.add_argument("--force", action="store_true",
                    help='Force.')
    args = ap.parse_args()

    name = args.name or f"traces-{args.probe}-s{args.seed}"
    out_dir = Path(args.out_dir) if args.out_dir else judge_samples_dir(args.run)
    out_rows, out_manifest = out_dir / f"{name}.jsonl", out_dir / f"{name}.manifest.json"
    if out_rows.exists() and not args.force:
        raise SystemExit(f"{out_rows} exists. A sample is frozen once drawn — judgments are keyed "
                         f"to it. Pick another --name, or pass --force to replace it.")

    paths = sorted(Path(p) for p in Path().glob(generations_glob(args.run)))
    if not paths:
        raise SystemExit(f"no generation files match {generations_glob(args.run)}")
    print(f"scanning {len(paths)} file(s) for think-on {args.probe} rows with a trace ...",
          flush=True)
    cells, sources, dupes = scan(paths, args.probe, set(args.models) if args.models else None)
    if not cells:
        raise SystemExit("no eligible rows: nothing is think-on with a reasoning trace here")
    picks, short = draw(cells, args.per_wave, args.baseline, args.seed)
    for key, want, got in short:
        print(f"  ⚠️  {_k(key)}: asked {want}, only {got} eligible — took all", flush=True)


    by_file = defaultdict(list)
    for key, item in picks:
        path, start, length, gid = cells[key][item]
        by_file[path].append((start, length, key, item, gid))
    lines, rows_meta = {}, {}
    for path, addrs in by_file.items():
        with open(path, "rb") as f:
            for start, length, key, item, gid in sorted(addrs):
                f.seek(start)
                raw = f.read(length)
                lines[(key, item)] = raw if raw.endswith(b"\n") else raw + b"\n"
                r = json.loads(raw)
                rows_meta[gid] = {"model": key[0], "concept_id": key[1], "condition": key[2],
                                  "prompt_id": key[3], "item_id": item, "source": path.name,
                                  "finish_reason": r.get("finish_reason"), **lexical_flags(r)}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_rows, "wb") as f:
        for k in sorted(lines):
            f.write(lines[k])

    manifest = {
        "run": args.run, "name": name, "probe": args.probe, "seed": args.seed,
        "per_wave": args.per_wave, "baseline": args.baseline, "models": args.models,
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "population": {_k(k): len(v) for k, v in sorted(cells.items())},
        "allocation": {_k(k): n for k, n in sorted(
            ((k, sum(1 for kk, _ in picks if kk == k)) for k in {kk for kk, _ in picks}))},
        "shortfalls": [{"stratum": _k(k), "asked": w, "eligible": g} for k, w, g in short],
        "duplicates_skipped": dupes, "rows": rows_meta, "sources": sources}
    out_manifest.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(lines):,} rows -> {out_rows}\nmanifest -> {out_manifest}")
    print(f"{'model':<22} {'concept':<11} {'supp':>5} {'base':>5} {'lex trace (novel)':>18} "
          f"{'trace chars p50/max':>20}")
    for model, concept in sorted({(m["model"], m["concept_id"]) for m in rows_meta.values()}):
        mine = [m for m in rows_meta.values() if (m["model"], m["concept_id"]) == (model, concept)]
        supp = sum(m["condition"] == "suppressed" for m in mine)
        nov = [m["lex_trace_novel"] for m in mine if m["lex_trace_novel"] is not None]
        chars = sorted(m["trace_chars"] for m in mine)
        print(f"{model:<22} {concept:<11} {supp:>5} {len(mine) - supp:>5} "
              f"{(f'{100 * sum(nov) / len(nov):.1f}%' if nov else '-'):>18} "
              f"{f'{chars[len(chars) // 2]:,}/{chars[-1]:,}':>20}")
    if dupes:
        print(f"({dupes} duplicate generation_id row(s) skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
