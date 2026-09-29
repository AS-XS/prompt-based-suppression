#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts  # noqa: E402
from src.analysis.surfaces import answer_of, explanation_of, trace_of  # noqa: E402
from src.data.schema import GenerationRow  # noqa: E402
from src.gen.env import constrain_to_schema, make_llm, setup_vllm_env  # noqa: E402
from src.gen.runner import append_jsonl, read_jsonl  # noqa: E402
from src.judge.prompts import judge_concept_text  # noqa: E402
from src.judge.run import run_primitive_judge  # noqa: E402
from src.judge.schema import judge_json_schema  # noqa: E402
from src.judge.settings import (default_budget, judge_fingerprint, judge_sampling,  # noqa: E402
                                refuse_mixed_config)
from src.paths import generations_glob, primitive_judgments_dir  # noqa: E402
from src.provenance import git_commit, select_one_config  # noqa: E402

_COLS = ["generation_id", "concept_id", "concept_class", "prompt_id", "condition", "think_state",
         "surface", "semantic_leak", "leak_type", "meta_reference", "fluent", "guessing",
         "parsed_ok", "judge"]


class _Row:


    def __init__(self, d: dict):
        self._d = d

    def model_dump_json(self) -> str:
        return json.dumps(self._d, ensure_ascii=False)


def _read_dicts(path: Path):
    if not path.exists():
        return
    yield from read_dicts(path)


def outstanding(gens_path, out_dir, suppressed_only, limit, surfaces, config_hash=None,
                shard=None, probe_types=None) -> int:


    rows = list(read_jsonl(Path(gens_path), GenerationRow))
    rows = select_one_config([r.model_dump() for r in rows], config_hash,
                             what=f"generations in {Path(gens_path).name}")
    rows = [GenerationRow.model_validate(r) for r in rows]
    if suppressed_only:
        rows = [r for r in rows if r.condition == "suppressed"]


    if probe_types:
        rows = [r for r in rows if r.probe_type in set(probe_types)]
    if limit:
        rows = rows[:limit]
    stem = Path(gens_path).stem
    if shard:
        i, n = shard
        rows = rows[i - 1::n]
        stem = f"{stem}.shard{i}of{n}"
    done = {(d.get("generation_id"), d.get("surface"))
            for d in _read_dicts(Path(out_dir) / f"{stem}.jsonl")}
    n = 0
    for r in rows:
        for surface in surfaces:
            if (r.generation_id, surface) in done:
                continue
            if surface == "answer":
                text = answer_of(r.raw_output)
            elif surface == "explanation":
                text = explanation_of(r.raw_output)
            else:
                text = trace_of(r.raw_output) or r.raw_reasoning
            if (text or "").strip():
                n += 1
    return n


def _judge_file(llm, sp, gens_path, out_dir, judge_name, suppressed_only, limit, surfaces,
                config_hash=None, shard=None, probe_types=None, fingerprint=None, meta=None):
    rows = list(read_jsonl(Path(gens_path), GenerationRow))
    rows = select_one_config([r.model_dump() for r in rows], config_hash,
                             what=f"generations in {Path(gens_path).name}")
    rows = [GenerationRow.model_validate(r) for r in rows]
    if suppressed_only:
        rows = [r for r in rows if r.condition == "suppressed"]
    if probe_types:
        rows = [r for r in rows if r.probe_type in set(probe_types)]
    if limit:
        rows = rows[:limit]

    stem = Path(gens_path).stem
    if shard:


        i, n = shard
        rows = rows[i - 1::n]
        stem = f"{stem}.shard{i}of{n}"
    out_jsonl = Path(out_dir) / f"{stem}.jsonl"
    out_csv = Path(out_dir) / f"{stem}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    prior = list(_read_dicts(out_jsonl))
    done = {(d.get("generation_id"), d.get("surface")) for d in prior}
    table = [{k: d.get(k) for k in _COLS} for d in prior]
    checked = False
    for r in rows:
        for surface in surfaces:
            if (r.generation_id, surface) in done:
                continue


            if surface == "answer":
                text = answer_of(r.raw_output)
            elif surface == "explanation":
                text = explanation_of(r.raw_output)
            else:
                text = trace_of(r.raw_output) or r.raw_reasoning
            if not (text or "").strip():
                continue
            if fingerprint and not checked:


                refuse_mixed_config(prior, fingerprint, out_jsonl)
                checked = True


            concept_text = judge_concept_text(r.concept_id)
            j = run_primitive_judge(llm, sp, concept=concept_text, sentence=text or "",
                                    surface=surface, concept_class=r.concept_class)
            rec = {"generation_id": r.generation_id, "model": r.model, "item_id": r.item_id,
                   "concept_id": r.concept_id, "concept_class": r.concept_class,
                   "probe_type": r.probe_type, "prompt_id": r.prompt_id,
                   "condition": r.condition, "think_state": r.think_state, "judge": judge_name,
                   "judge_concept": concept_text, **(meta or {}), **j}
            append_jsonl(out_jsonl, [_Row(rec)])
            table.append({k: rec.get(k) for k in _COLS})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    return len(table), out_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--generations", nargs="+", default=None, help='Generations.')
    ap.add_argument("--judge", required=True, help='Judge.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quant", default=None, help='Quant.')
    ap.add_argument("--tp", type=int, default=None, metavar="N",
                    help='Tp.')
    ap.add_argument("--dtype", default=None,
                    help='Dtype.')
    ap.add_argument("--suppressed-only", action="store_true")
    ap.add_argument("--probe-types", nargs="+", default=None,
                    choices=["forget", "neighbor", "retain", "utility", "dependent"],
                    help='Probe types.')
    ap.add_argument("--surfaces", nargs="+", default=["explanation", "trace"],
                    choices=["answer", "explanation", "trace"],
                    help='Surfaces.')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help='Shard.')
    ap.add_argument("--config-hash", default=None,
                    help='Config hash.')
    ap.add_argument("--max-tokens", type=int, default=None,
                    help='Max tokens.')
    ap.add_argument("--max-model-len", type=int, default=None,
                    help='Max model len.')
    ap.add_argument("--gpu-util", type=float, default=None,
                    help='Gpu util.')
    args = ap.parse_args()

    if not args.run and not args.generations:
        raise SystemExit("pass --run exp1|round1 (or --generations explicitly)")
    gen_globs = args.generations or [generations_glob(args.run)]
    out_dir = args.out_dir or primitive_judgments_dir(args.run)
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched: {gen_globs}")

    shard = None
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must look like i/N (got {args.shard!r})")
        if not (1 <= i <= n):
            raise SystemExit(f"--shard i must be within 1..N (got {args.shard!r})")
        shard = (i, n)


    todo = sum(outstanding(p, out_dir, args.suppressed_only, args.limit, args.surfaces,
                           args.config_hash, shard, args.probe_types) for p in paths)
    if not todo:
        print(f"nothing outstanding across {len(paths)} file(s) — NOT loading the judge, no GPU used.")
        return 0
    print(f"{todo} judgment(s) outstanding across {len(paths)} file(s)")


    fingerprint = judge_fingerprint("primitive", args.judge, quant=args.quant, tp=args.tp,
                                    dtype=args.dtype, max_tokens=args.max_tokens,
                                    max_model_len=args.max_model_len)
    from src.gen.thinkstate import needs_raw_special
    needs_raw_special(args.judge)
    model_dir = Path(args.models_dir) / args.judge
    budget = args.max_tokens if args.max_tokens is not None else default_budget(model_dir)

    setup_vllm_env()
    kwargs = {k: v for k, v in (("quantization", args.quant), ("tensor_parallel_size", args.tp),
                                ("dtype", args.dtype), ("max_model_len", args.max_model_len),
                                ("gpu_memory_utilization", args.gpu_util)) if v}
    if args.tp and args.quant:
        print("  note: --tp with --quant is unusual — 4-bit already fits one Ampere card, and "
              "sharding it adds communication for no memory win. Drop --quant if the point is to "
              "escape the Ampere requirement.")
    print(f"loading judge {args.judge}{' (' + args.quant + ')' if args.quant else ''}"
          f"{f' tp={args.tp}' if args.tp else ''} "
          f"for {len(paths)} file(s), surfaces={args.surfaces}, config {fingerprint}")
    llm = make_llm(str(model_dir), **kwargs)
    sp, resolved = judge_sampling(llm, args.judge, budget)
    sp = constrain_to_schema(sp, judge_json_schema())
    print("  judge decode (the model's own): " + ", ".join(f"{k}={v}" for k, v in resolved.items()),
          flush=True)
    meta = {"judge_config_hash": fingerprint, "judge_sampling": resolved,
            "git_commit": git_commit()}

    total = 0
    for p in paths:
        n, out = _judge_file(llm, sp, p, out_dir, args.judge,
                             args.suppressed_only, args.limit, args.surfaces,
                             args.config_hash, shard, args.probe_types, fingerprint, meta)
        total += n
        print(f"  {Path(p).name}: {n} primitive judgments -> {out}")
    print(f"done — {total} primitive judgments across {len(paths)} file(s); "
          f"feed to: python scripts/mode_table.py --run {args.run or '<run>'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
