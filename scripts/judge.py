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

from src.analysis.surfaces import answer_of, explanation_of, trace_of  # noqa: E402
from src.data.schema import GenerationRow  # noqa: E402
from src.gen.env import make_llm, setup_vllm_env  # noqa: E402
from src.gen.runner import append_jsonl, read_jsonl  # noqa: E402
from src.judge.run import (judge_chat_kwargs, run_concept, run_framing,  # noqa: E402
                           run_judge)
from src.paths import generations_glob, judgments_dir  # noqa: E402


def _surface_text(row, surface: str):


    if surface == "answer":
        return answer_of(row.raw_output, expect_trace=row.think_state == "on")
    if surface == "explanation":
        return explanation_of(row.raw_output)
    return trace_of(row.raw_output) or row.raw_reasoning

_COLS = ["generation_id", "concept_id", "concept_class", "prompt_id", "condition", "think_state",
         "surface", "concept_score", "framing_score"]


def _read_dicts(path: Path):

    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def plan_file(gens_path, out_dir, suppressed_only, limit, surfaces=("explanation",),
              shard=None, stage="both", probe_types=None) -> dict:


    rows = list(read_jsonl(Path(gens_path), GenerationRow))
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
    best: dict[tuple, dict] = {}
    for d in prior:
        k = (d.get("generation_id"), d.get("surface"))

        if k not in best or (d.get("framing_score") is not None):
            best[k] = d
    table = [{k: d.get(k) for k in _COLS} for d in best.values()]
    todo = []
    for r in rows:
        for surface in surfaces:
            key = (r.generation_id, surface)
            done = best.get(key)
            if stage == "concept" and done is not None:
                continue
            if stage == "framing":
                if done is None:
                    continue
                c = done.get("concept_score")
                if not c or c <= 0 or done.get("framing_score") is not None:
                    continue
            if stage == "both" and done is not None:
                continue
            text = _surface_text(r, surface)
            if not (text or "").strip():
                continue
            todo.append((r, surface, text, done))
    return {"stem": stem, "out_jsonl": out_jsonl, "out_csv": out_csv, "best": best,
            "table": table, "todo": todo}


def write_csv(plan: dict) -> None:

    with plan["out_csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        w.writerows(plan["table"])


def run_plan(llm, sp, plan: dict, judge_name, stage="both", chat_kw=None) -> tuple[int, Path]:


    best, table, out_jsonl = plan["best"], plan["table"], plan["out_jsonl"]
    written = 0
    for r, surface, text, done in plan["todo"]:
        key = (r.generation_id, surface)
        if stage == "framing":


            j = {**done, **run_framing(llm, sp, concept=r.concept_id, sentence=text,
                                       chat_kw=chat_kw)}
        elif stage == "concept":
            j = run_concept(llm, sp, concept=r.concept_id, sentence=text, chat_kw=chat_kw)
        else:
            j = run_judge(llm, sp, concept=r.concept_id, sentence=text, chat_kw=chat_kw)
        rec = {"generation_id": r.generation_id, "concept_id": r.concept_id,
               "concept_class": r.concept_class, "prompt_id": r.prompt_id,
               "condition": r.condition, "think_state": r.think_state,
               "surface": surface, "judge": judge_name, **j}
        append_jsonl(out_jsonl, [_Row(rec)])
        best[key] = rec
        table[:] = [t for t in table if (t.get("generation_id"), t.get("surface")) != key]
        table.append({k: rec.get(k) for k in _COLS})
        written += 1

    write_csv(plan)
    return written, out_jsonl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--generations", nargs="+", default=None,
                    help='Generations.')
    ap.add_argument("--judge", required=True, help='Judge.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quant", default=None, help='Quant.')
    ap.add_argument("--tp", type=int, default=None, metavar="N",
                    help='Tp.')
    ap.add_argument("--dtype", default=None, help='Dtype.')
    ap.add_argument("--suppressed-only", action="store_true")
    ap.add_argument("--probe-types", nargs="+", default=None,
                    choices=["forget", "neighbor", "retain", "utility", "dependent"],
                    help='Probe types.')
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=2000,
                    help='Max tokens.')
    ap.add_argument("--surfaces", nargs="+", default=["explanation"],
                    choices=["answer", "explanation", "trace"],
                    help='Surfaces.')
    ap.add_argument("--stage", default="both", choices=["concept", "framing", "both"],
                    help='Stage.')
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help='Shard.')
    args = ap.parse_args()

    shard = None
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must look like i/N (got {args.shard!r})")
        if not (1 <= i <= n):
            raise SystemExit(f"--shard i must be within 1..N (got {args.shard!r})")
        shard = (i, n)

    if not args.run and not args.generations:
        raise SystemExit("pass --run exp1|round1 (or --generations explicitly)")
    gen_globs = args.generations or [generations_glob(args.run)]
    out_dir = args.out_dir or judgments_dir(args.run)
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no files matched: {gen_globs}")


    plans = [plan_file(p, out_dir, args.suppressed_only, args.limit, args.surfaces, shard,
                       args.stage, args.probe_types) for p in paths]
    outstanding = sum(len(pl["todo"]) for pl in plans)
    for pl in plans:
        write_csv(pl)
    if not outstanding:
        done = sum(len(pl["table"]) for pl in plans)
        print(f"nothing outstanding for stage={args.stage} across {len(paths)} file(s) "
              f"({done} judgments already on disk) — NOT loading the judge, no GPU used.")
        return 0
    print(f"{outstanding} judgment(s) outstanding across {len(paths)} file(s)")

    setup_vllm_env()
    kwargs = {"max_model_len": 8192}
    if args.quant:
        kwargs["quantization"] = args.quant
    if args.tp:
        kwargs["tensor_parallel_size"] = args.tp
    if args.dtype:
        kwargs["dtype"] = args.dtype
    print(f"loading judge {args.judge}{' (' + args.quant + ')' if args.quant else ''}"
          f"{f' tp={args.tp}' if args.tp else ''} "
          f"for {len(paths)} file(s), surfaces={args.surfaces}, stage={args.stage}")
    llm = make_llm(str(Path(args.models_dir) / args.judge), **kwargs)
    from vllm import SamplingParams


    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    chat_kw = judge_chat_kwargs(args.judge)
    if chat_kw:
        print(f"  judge reasoning disabled via {chat_kw} (it must answer, not deliberate)")

    total = 0
    for pl in plans:
        if not pl["todo"]:
            continue
        n, out = run_plan(llm, sp, pl, args.judge, args.stage, chat_kw)
        total += n
        print(f"  {pl['stem']}: {n} judgments -> {out}")
    print(f"done — {total} judgments across {len(paths)} file(s)")
    return 0


class _Row:


    def __init__(self, d: dict):
        self._d = d

    def model_dump_json(self) -> str:
        import json

        return json.dumps(self._d, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
