from __future__ import annotations

import re
from pathlib import Path

from src.data.schema import GenerationRow, Item, Sampling
from src.gen.env import make_llm
from src.gen.runner import append_jsonl, read_jsonl
from src.gen.thinkstate import chat_kwargs, needs_raw_special, think_states_for
from src.prompts import (DEFAULT_ANSWER_FORMAT, build_messages, draw_all, is_directed_prompt,
                         prompt_fingerprint, render_question, slots_for, split_prompt_id,
                         takes_examples)
from src.provenance import config_hash, git_commit


_BASELINE_PROMPT = "none-baseline"


_SHARD_SUFFIX = re.compile(r"\.shard\d+of\d+$")


def _existing_ids(path: Path) -> set[str]:


    stem = _SHARD_SUFFIX.sub("", path.stem)
    ids: set[str] = set()
    for sibling in sorted(path.parent.glob(f"{stem}*.jsonl")):
        ids |= {r.generation_id for r in read_jsonl(sibling, GenerationRow)}
    return ids


DEFAULT_CHUNK = 1000


MIN_CONTEXT = 8192
PROMPT_HEADROOM = 4096


def context_for(sampling: Sampling) -> int:


    return max(MIN_CONTEXT, int(sampling.max_tokens) + PROMPT_HEADROOM)


def model_context_limit(model_path: str | Path) -> int | None:


    import json

    cfg_file = Path(model_path) / "config.json"
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("max_position_embeddings", "n_positions", "max_seq_len"):
        for holder in (cfg, cfg.get("text_config") or {}):
            v = holder.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return int(v)
    return None


def clamp_context(want: int, model_path: str | Path, sampling: Sampling) -> tuple[int, str | None]:


    limit = model_context_limit(model_path)
    if limit is None or want <= limit:
        return want, None
    room = limit - PROMPT_HEADROOM
    msg = (f"  context: model supports {limit:,}; clamping {want:,} -> {limit:,}. "
           f"Effective answer budget is {limit:,} minus the prompt "
           f"(~{max(room, 0):,} on a {PROMPT_HEADROOM:,}-token prompt) against "
           f"max_tokens={int(sampling.max_tokens):,}.")
    if sampling.max_tokens > room:
        msg += (f"\n  ⚠️  max_tokens exceeds that room, so long-prompt items generate less than "
                f"asked. Fine when the cap never binds (check truncated_pct); otherwise lower "
                f"max_tokens or drop this model from the run.")
    return limit, msg


def make_engine(model_name: str, model_path: str, sampling: Sampling,
                max_model_len: int | None = None, tp: int = 1,
                gpu_util: float = 0.85):


    extra = {"tensor_parallel_size": tp} if tp > 1 else {}
    llm = make_llm(model_path, dtype="auto", gpu_memory_utilization=gpu_util,
                   max_model_len=max_model_len or context_for(sampling), **extra)
    from vllm import SamplingParams


    opt = {k: v for k, v in (("temperature", sampling.temperature), ("top_p", sampling.top_p),
                             ("seed", sampling.seed),
                             ("repetition_penalty", sampling.repetition_penalty))
           if v is not None}
    sp = SamplingParams(max_tokens=sampling.max_tokens,
                        skip_special_tokens=not needs_raw_special(model_name), **opt)
    resolved = {k: getattr(sp, k, None)
                for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                          "presence_penalty", "frequency_penalty", "max_tokens", "seed", "n")}
    print(f"  decode ({model_name}): " + ", ".join(f"{k}={v}" for k, v in resolved.items()
                                                   if v is not None), flush=True)
    return llm, sp, resolved


def run_sweep(model_name: str, model_path: str, items: list[Item], out_path: str | Path, *,
              conditions: list[str], suppressed_prompt: str, sampling: Sampling, llm=None,
              sampling_params=None, max_model_len: int | None = None, tp: int = 1,
              gpu_util: float = 0.85,
              answer_format: str | None = None,
              chunk_size: int = DEFAULT_CHUNK) -> int:


    out_path = Path(out_path)


    cfg = {"model": model_name, "suppressed_prompt": suppressed_prompt,
           "prompt_fingerprint": prompt_fingerprint(suppressed_prompt, answer_format=answer_format),


           "sampling": sampling.model_dump(exclude_none=True)}
    chash, commit = config_hash(cfg), git_commit()
    fmt = answer_format or DEFAULT_ANSWER_FORMAT
    states = think_states_for(model_name)


    examples_by_concept: dict[str, list[tuple[str, str]]] = {}
    if is_directed_prompt(split_prompt_id(suppressed_prompt)[0]) and \
            takes_examples(split_prompt_id(suppressed_prompt)[0]):
        template_id = split_prompt_id(suppressed_prompt)[0]
        examples_by_concept, held_out = draw_all(
            items, [it.concept_id for it in items], prompt_id=template_id, answer_format=fmt)
        if not examples_by_concept:
            tiers = ", ".join(p for p, _ in slots_for(template_id))
            raise SystemExit(
                f"{suppressed_prompt!r} needs one MC item per concept in each of [{tiers}] to "
                f"build its examples, and the item set has none. Check probe_types.")
        items = [it for it in items if it.item_id not in held_out]
        print(f"  examples: {sum(len(v) for v in examples_by_concept.values())} drawn over "
              f"{len(examples_by_concept)} concept(s); {len(held_out)} item(s) held out", flush=True)
    done = _existing_ids(out_path)


    cells = []
    for it in items:
        for cond in conditions:
            prompt_id = _BASELINE_PROMPT if cond == "baseline" else suppressed_prompt
            for ts in states:
                if f"{chash}:{it.item_id}:{cond}:{ts}" not in done:
                    cells.append((it, cond, prompt_id, ts))
    if not cells:
        return 0

    if llm is None:
        llm, sp, resolved = make_engine(model_name, model_path, sampling,
                                        max_model_len=max_model_len, tp=tp, gpu_util=gpu_util)
    else:


        resolved = {k: getattr(sampling_params, k, None)
                    for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                              "presence_penalty", "frequency_penalty", "max_tokens", "seed", "n")
                    } if sampling_params is not None else None


        sp = sampling_params if sampling_params is not None else sampling

    written = 0
    for ts in states:
        batch = [c for c in cells if c[3] == ts]
        if not batch:
            continue
        for start in range(0, len(batch), chunk_size):
            part = batch[start:start + chunk_size]


            messages = [build_messages(pid, it.concept_id, render_question(it, fmt),
                                       examples_by_concept.get(it.concept_id))
                        for (it, _, pid, _) in part]
            outs = llm.chat(messages, sp, **chat_kwargs(model_name, ts))
            rows = []
            for (it, cond, pid, _), out in zip(part, outs):
                o = out.outputs[0]
                rows.append(GenerationRow(
                    git_commit=commit, config_hash=chash, config_id=model_name, model=model_name,
                    prompt_id=pid, condition=cond, think_state=ts, sampling=sampling,
                    resolved_sampling=resolved,
                    item_id=it.item_id, concept_id=it.concept_id, concept_class=it.concept_class,
                    concept_subtype=it.concept_subtype, probe_type=it.probe_type,
                    raw_output=o.text, raw_reasoning=None, finish_reason=o.finish_reason,
                    completion_tokens=len(o.token_ids), prompt_tokens=len(out.prompt_token_ids),
                ))
            written += append_jsonl(out_path, rows)

            print(f"  [{model_name} {ts}] {min(start + chunk_size, len(batch)):,}/{len(batch):,} "
                  f"({written:,} rows written)", flush=True)
    return written
