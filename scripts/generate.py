#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.data.schema import Item, Sampling  # noqa: E402
from src.gen.runner import read_jsonl  # noqa: E402
from src.gen.sweep import run_sweep  # noqa: E402
from src.gen.thinkstate import think_states_for  # noqa: E402
from src.paths import generations_dir, items_path  # noqa: E402
from src.prompts import ANSWER_FORMATS, DEFAULT_ANSWER_FORMAT  # noqa: E402


def load_items(path: Path, concepts, probe_types, max_items, sample: int | None = None) -> list[Item]:
    items = list(read_jsonl(path, Item))
    if concepts:
        items = [it for it in items if it.concept_id in set(concepts)]
    if probe_types:
        items = [it for it in items if it.probe_type in set(probe_types)]
    if sample:


        import random

        return random.Random(0).sample(items, min(sample, len(items)))
    return items[:max_items] if max_items else items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None, help='Model.')
    ap.add_argument("--prompt-suppressed", nargs="+", default=None,
                    help='Prompt suppressed.')
    ap.add_argument("--conditions", nargs="+", default=None,
                    help='Conditions.')
    ap.add_argument("--run", default=None,
                    help='Run.')
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--out-dir", default=None, help='Out dir.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--gpu-util", type=float, default=float(os.environ.get("GPU_UTIL", "0.85")),
                    help='Gpu util.')
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "1")),
                    help='Tp.')
    ap.add_argument("--limit", type=int, default=None,
                    help='Limit.')
    ap.add_argument("--sample", type=int, default=None,
                    help='Sample.')
    ap.add_argument("--shard", default=None, metavar="i/N",
                    help='Shard.')
    ap.add_argument("--chunk-size", type=int, default=None,
                    help='Chunk size.')
    ap.add_argument("--max-tokens", type=int, default=None,
                    help='Max tokens.')
    args = ap.parse_args()


    shard, shard_suffix = None, ""
    if args.shard:
        try:
            si, sn = (int(x) for x in args.shard.split("/"))
        except ValueError:
            raise SystemExit(f"--shard must look like i/N (got {args.shard!r})")
        if not (1 <= si <= sn):
            raise SystemExit(f"--shard i must be within 1..N (got {args.shard!r})")
        shard, shard_suffix = (si, sn), f".shard{si}of{sn}"

    cfg = yaml.safe_load(Path(args.config).read_text())
    model = args.model or cfg.get("model")
    if not model:
        raise SystemExit("no model: set --model or `model:` in the config")


    answer_format = cfg.get("answer_format", DEFAULT_ANSWER_FORMAT)
    if answer_format not in ANSWER_FORMATS:
        raise SystemExit(f"unknown answer_format {answer_format!r} in {args.config}; "
                         f"expected one of {ANSWER_FORMATS}")


    run = args.run or cfg.get("run")
    if not run and not (args.items and args.out_dir):
        raise SystemExit(f"no run namespace: add `run: exp1|round1` to {args.config} (or pass "
                         "--run, or both --items and --out-dir)")
    items_file = Path(args.items) if args.items else items_path(run)
    out_root = Path(args.out_dir) if args.out_dir else generations_dir(run)
    conditions = args.conditions or cfg["conditions"]
    prompts = args.prompt_suppressed or [cfg.get("prompt_suppressed", "world-frame")]
    if len(prompts) > 1 and len(conditions) > 1:


        raise SystemExit(
            f"--prompt-suppressed takes several ids only with a single condition (got "
            f"conditions={conditions}). Run the baseline as its own wave and the ladder as another:\n"
            f"  --conditions baseline\n"
            f"  --conditions suppressed --prompt-suppressed {' '.join(prompts)}")
    if not items_file.exists():
        raise SystemExit(f"no item set at {items_file} — build it first:\n"
                         f"  python -m src.data.build --run {run} ...")
    items = load_items(items_file, cfg.get("concepts"), cfg.get("probe_types"),
                       args.limit or cfg.get("max_items"), sample=args.sample)
    if shard:


        si, sn = shard
        items = items[si - 1::sn]


    samp = dict(cfg["sampling"])
    if args.max_tokens:
        samp["max_tokens"] = args.max_tokens
    sampling = Sampling(**samp)
    model_path = str(Path(args.models_dir) / model)
    from src.gen.sweep import DEFAULT_CHUNK, clamp_context, context_for, make_engine


    max_model_len = cfg.get("max_model_len") or context_for(sampling)


    max_model_len, clamp_msg = clamp_context(max_model_len, model_path, sampling)
    if clamp_msg:
        print(clamp_msg, flush=True)
    llm = sp = None
    if len(prompts) > 1:


        llm, sp, _resolved = make_engine(model, model_path, sampling,
                                         max_model_len=max_model_len,
                                         tp=args.tp, gpu_util=args.gpu_util)

    total = 0
    for suppressed_prompt in prompts:
        wave = "-".join(sorted(conditions)) if len(conditions) > 1 else (
            "none-baseline" if conditions == ["baseline"] else suppressed_prompt)
        out_path = out_root / f"{model}__{wave}{shard_suffix}.jsonl"
        print(f"[{run}] {model}: {len(items)} items × {conditions} × prompt[{suppressed_prompt}] × "
              f"think{think_states_for(model)} × fmt[{answer_format}] -> {out_path}", flush=True)
        n = run_sweep(model, model_path, items, out_path,
                      conditions=conditions, suppressed_prompt=suppressed_prompt,
                      sampling=sampling, llm=llm, sampling_params=sp,
                      max_model_len=max_model_len, tp=args.tp, gpu_util=args.gpu_util,
                      answer_format=answer_format,
                      chunk_size=args.chunk_size or DEFAULT_CHUNK)
        print(f"wrote {n} new rows (resumable — re-run to continue)", flush=True)
        total += n
    if len(prompts) > 1:
        print(f"[{run}] {model}: {len(prompts)} waves done, {total:,} new rows total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
