#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.opt import result_path  # noqa: E402
from src.opt.adapter import VerbatimAdapter  # noqa: E402
from src.opt.clause import (best_wellformed, looks_like_meta, meta_reason,  # noqa: E402
                            restates_instrument)
from src.opt.dataset import build_examples, stratified_split  # noqa: E402
from src.opt.metrics import OBJECTIVES, PROBE_TYPES, metric_for  # noqa: E402
from src.opt.program import build_program, clause_of, clause_signature, seed_clause  # noqa: E402
from src.opt.serving import check_think_state, extra_body  # noqa: E402
from src.paths import (generations_dir, items_path, optimize_namespace,  # noqa: E402
                       split_path)
from src.provenance import git_commit  # noqa: E402


def _default_trials(n_candidates: int, n_predictors: int = 1) -> int:


    import math

    return int(max(2 * n_predictors * math.log2(max(n_candidates, 2)), 1.5 * n_candidates))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="wmdp1",
                    help='Run.')
    ap.add_argument("--round", type=int, default=1, metavar="N",
                    help='Round.')
    ap.add_argument("--out-run", default=None,
                    help='Out run.')
    ap.add_argument("--force", action="store_true",
                    help='Force.')
    ap.add_argument("--model", required=True, help='Model.')
    ap.add_argument("--objective", default="A", choices=OBJECTIVES)
    ap.add_argument("--seed-prompt", default="scoped-world@generic",
                    help='Seed prompt.')
    ap.add_argument("--concept", nargs="+", default=["wmdp-bio"],
                    help='Concept.')
    ap.add_argument("--think-state", default="off")
    ap.add_argument("--limit", type=int, default=None, help='Limit.')
    ap.add_argument("--candidates", type=int, default=12,
                    help='Candidates.')
    ap.add_argument("--trials", type=int, default=None,
                    help='Trials.')
    ap.add_argument("--minibatch", action="store_true",
                    help='Minibatch.')
    ap.add_argument("--demos", type=int, default=0, metavar="N",
                    help='Demos.')
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--prompt-model", default=None,
                    help='Prompt model.')
    ap.add_argument("--prompt-model-api-base", default="http://localhost:8001/v1")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help='Max tokens.')
    ap.add_argument("--dry-run", action="store_true",
                    help='Dry run.')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()


    if not args.prompt_model and args.think_state == "on":
        raise SystemExit(
            f"--think-state on with no --prompt-model: the target would propose its own clauses "
            f"while thinking, and its trace collides with DSPy's field markers — every proposal "
            f"fails to parse and the search dies without writing.\n"
            f"  Either pass PROMPT_MODEL=<a model served thinking-off> (D16, and the Llama "
            f"self-proposal result says to anyway: 7 of 19 candidates were meta-text),\n"
            f"  or run the target with --think-state off.")


    import dspy

    items = items_path(args.run)
    baseline = generations_dir(args.run) / f"{args.model}__none-baseline.jsonl"
    for p in (items, baseline, split_path(args.run)):
        if not p.exists():
            raise SystemExit(f"missing {p} — the DSPy arm needs the item set, the frozen split, and "
                             f"this model's own baseline wave (it is what every metric pairs against)")

    probe_types = PROBE_TYPES[args.objective]
    examples = build_examples(items, baseline, split_path(args.run), think_state=args.think_state,
                              concepts=tuple(args.concept), probe_types=probe_types,
                              limit=args.limit)
    if not examples:
        raise SystemExit(
            "0 examples. Every one of these silently produces an empty set: the baseline wave is for "
            "a different think-state, the concept has no val items, or the model got nothing right "
            "(examples the baseline failed are dropped — there is nothing there to suppress).")

    from collections import Counter
    mix = Counter(e.probe_type for e in examples)
    print(f"example mix: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items())))
    if args.objective == "A" and set(mix) == {"forget"}:
        raise SystemExit(
            "objective A has ONLY forget examples, so its `intact` term is unreachable and it is "
            "identical to objective B — the two would be the same experiment run twice (this "
            "happened). Check that the run's retain arms reached build_examples.")


    missing = [p for p in probe_types if p not in mix]
    if missing:
        print(f"WARNING: probe arm(s) {missing} were requested for objective {args.objective} but "
              f"contributed ZERO examples. Objective A's `intact` term is measured only over the "
              f"arms that are present, so its retain claim is narrower than it looks. Most likely "
              f"cause: those probes are open-ended, and build_examples keeps MC items only.",
              file=sys.stderr)


    trainset, valset = stratified_split(examples)
    dev_mix = Counter(e.probe_type for e in valset)
    print(f"dev mix: " + ", ".join(f"{k} {v}" for k, v in sorted(dev_mix.items())))
    if args.objective == "A" and "forget" not in dev_mix:
        raise SystemExit("objective A's dev set has no forget examples — its score would measure "
                         "only the retain arm. Check "
                         "stratified_split.")


    body = extra_body(args.model, args.think_state)
    lm = dspy.LM(f"openai/{args.model}", api_base=args.api_base, api_key="EMPTY",
                 temperature=0.0, max_tokens=args.max_tokens, cache=False,
                 **({"extra_body": body} if body else {}))
    dspy.configure(lm=lm, adapter=VerbatimAdapter())

    seed = seed_clause(args.seed_prompt, args.concept[0])
    program = build_program(seed)
    metric = metric_for(args.objective)
    frozen_desc = (clause_signature(program).output_fields["response"].json_schema_extra or {}).get("desc")

    print(f"[{args.run}] {args.model} · objective {args.objective} · {len(examples)} val examples "
          f"({len(trainset)} train / {len(valset)} dev) · think={args.think_state}")
    print(f"seed clause ({args.seed_prompt}):\n  {seed[:300]}")


    probe = program(question=examples[0].question)
    why = check_think_state(args.model, args.think_state, getattr(probe, "response", "") or "")
    if why:
        raise SystemExit(f"think-state preflight FAILED — {why}\n"
                         f"  requested think_state={args.think_state!r}, extra_body={body}\n"
                         f"Do not use any numbers from this configuration.")
    print(f"think-state preflight ok (extra_body={body or 'none needed'})")


    longest = max(examples, key=lambda e: len(e.question))


    probe_prog = program
    if args.demos:
        probe_prog = build_program(seed)
        probe_prog.demos = sorted(trainset, key=lambda e: -len(e.question))[:args.demos]
        print(f"context preflight: probing with {len(probe_prog.demos)} demo(s) attached "
              f"(the {args.demos} longest train items)")
    try:
        probe_prog(question=longest.question)
    except Exception as e:
        if "context" in str(e).lower() or "maximum context length" in str(e).lower():
            raise SystemExit(
                f"context preflight FAILED on the longest dev item ({len(longest.question)} chars):\n"
                f"  {e}\n"
                f"The server would 400 on this item and DSPy would score it 0 rather than raise, so "
                f"the run would finish with a plausible mean over silently-failed items — and would "
                f"favour SHORT clauses, since a longer one pushes more items over the limit.\n"
                f"Fix: raise the served context (`MAX_MODEL_LEN=...`, capped at the model's own "
                f"limit), lower --max-tokens (currently {args.max_tokens})"
                + (f", or lower --demos (currently {args.demos} — each demo prepends a whole "
                   f"question and answer)." if args.demos else "."))
        raise
    print(f"context preflight ok (longest dev item {len(longest.question)} chars "
          f"+ {args.max_tokens} output tokens)")

    evaluate = dspy.Evaluate(devset=valset, metric=metric, num_threads=8,
                             display_progress=True, display_table=0)


    n_trials_req = args.trials if args.trials is not None else _default_trials(args.candidates)
    print(f"search budget: {args.candidates} candidates x {n_trials_req} trials, "
          f"minibatch={args.minibatch} over {len(valset)} dev items")

    def per_item(prog) -> dict[str, int]:


        out = {}
        for ex in valset:
            out[ex.item_id] = int(metric(ex, prog(question=ex.question)))
        return out

    t0 = time.time()
    seed_score = float(evaluate(program).score)
    seed_items = per_item(program)
    print(f"\nseed score: {seed_score:.4f}  ({time.time() - t0:.0f}s)")


    out_run = args.out_run or optimize_namespace(
        args.round, prompt_model=args.prompt_model, demos=args.demos)
    out = (Path(args.out) if args.out
           else result_path(out_run, args.model, args.objective, args.dry_run))
    out.parent.mkdir(parents=True, exist_ok=True)

    def _write(payload: dict) -> None:


        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(out)

    _write({"status": "seed-only — the search did not finish; this is NOT a result",
            "git_commit": git_commit(), "run": args.run, "out_run": out_run,
            "model": args.model, "objective": args.objective, "round": args.round,
            "seed_prompt": args.seed_prompt, "concepts": list(args.concept),
            "think_state": args.think_state, "prompt_model": args.prompt_model,
            "demos": args.demos, "n_dev": len(valset), "n_train": len(trainset),


            "max_tokens": args.max_tokens, "minibatch": args.minibatch, "dev_mix": dict(dev_mix),
            "candidates": args.candidates, "trials": n_trials_req,
            "seed_clause": seed, "seed_score": seed_score, "seed_items": seed_items})
    print(f"  banked seed-only result -> {out}")


    found, found_score, found_items = seed, seed_score, seed_items
    search: dict = {}
    if not args.dry_run:
        from dspy.teleprompt import MIPROv2


        prompt_lm = None
        if args.prompt_model:


            prompt_lm = dspy.LM(f"openai/{args.prompt_model}", api_base=args.prompt_model_api_base,
                                api_key="EMPTY", temperature=1.0, max_tokens=1024, cache=False,
                                **({"extra_body": extra_body(args.prompt_model, "off")}
                                   if extra_body(args.prompt_model, "off") else {}))
            try:
                probe_p = prompt_lm("Reply with the single word OK.")
            except Exception as e:
                raise SystemExit(
                    f"--prompt-model {args.prompt_model} is not reachable at "
                    f"{args.prompt_model_api_base}: {e}\n"
                    f"Start the proposer server on its own allocated GPU and check its endpoint.")
            print(f"proposer: {args.prompt_model} at {args.prompt_model_api_base} "
                  f"(probe: {str(probe_p)[:40]!r})")


        n_demos = args.demos
        if n_demos:


            missing = [e.item_id for e in trainset if not getattr(e, "response", None)]
            if missing:
                raise SystemExit(
                    f"--demos {n_demos} but {len(missing)} of {len(trainset)} train examples carry "
                    f"no `response` (e.g. {missing[:3]}). Labeled demos are drawn from the "
                    f"trainset, so every example needs one — see opt/dataset.py::build_examples. "
                    f"Without this check the run would SUCCEED and silently return a no-demo prompt.")
            print(f"few-shot: up to {n_demos} demo(s) per candidate (labeled + bootstrapped; "
                  f"DSPy cannot disable bootstrapping above 3 demo sets — see the comment above)")
        opt = MIPROv2(metric=metric, auto=None, num_candidates=args.candidates,
                      max_bootstrapped_demos=n_demos, max_labeled_demos=n_demos,
                      num_threads=8, seed=0,
                      verbose=False, **({"prompt_model": prompt_lm} if prompt_lm else {}))
        compiled = opt.compile(program, trainset=trainset, valset=valset, num_trials=n_trials_req,
                               max_bootstrapped_demos=n_demos, max_labeled_demos=n_demos, seed=0,
                               minibatch=args.minibatch)
        found = clause_of(compiled)
        found_score = float(evaluate(compiled).score)
        found_items = per_item(compiled)


        cands = getattr(compiled, "candidate_programs", []) or []
        clauses = {c["program"].signature.instructions for c in cands if isinstance(c, dict)
                   and hasattr(getattr(c.get("program"), "signature", None), "instructions")}


        alt, alt_score, n_meta = best_wellformed(cands)
        if looks_like_meta(found):
            print(f"\n⚠️  the winning clause is META-TEXT — it {meta_reason(found)}.")
            print("    Prepended to the user turn this asks the model to DESIGN a prompt, so the")
            print("    score describes a malformed input, not a suppression wording.")
            if alt:
                print(f"    Best well-formed candidate scored {alt_score:.2f}: {alt[:120]}…")
            else:
                print("    NO candidate was well-formed — this run produced no usable prompt.")


        candidates = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            prog = c.get("program")
            txt = getattr(getattr(prog, "signature", None), "instructions", None)
            if not txt:
                continue
            demos = list(getattr(prog, "demos", []) or [])
            candidates.append({
                "clause": txt,


                "n_demos": len(demos),
                "demo_answers": [str(getattr(d, "response", "") or
                                     (d.get("response") if hasattr(d, "get") else ""))[:60]
                                 for d in demos],
                "score": (round(float(c["score"]), 4) if c.get("score") is not None else None),
                "is_meta": looks_like_meta(txt),
                "restates_instrument": restates_instrument(txt, seed),
                "is_seed": txt == seed,
            })


        clean = [c for c in candidates
                 if c["score"] is not None and not c["is_meta"]
                 and not c["restates_instrument"] and not c["is_seed"]]
        best_clean = max(clean, key=lambda c: c["score"]) if clean else None
        if best_clean:
            print(f"\nbest candidate that is neither meta-text NOR a restatement of the instrument: "
                  f"{best_clean['score']:.2f} (seed {seed_score:.2f})")
        else:
            print("\nNO candidate was both well-formed and free of instrument restatement.")

        search = {
            "candidates": candidates,
            "n_clean_candidates": len(clean),
            "best_clean_score": best_clean["score"] if best_clean else None,
            "best_clean_clause": best_clean["clause"] if best_clean else None,
            "n_candidates": len(cands),
            "found_is_meta": looks_like_meta(found),
            "meta_reason": meta_reason(found) if looks_like_meta(found) else "",
            "n_meta_candidates": n_meta,
            "best_wellformed_clause": alt,
            "best_wellformed_score": alt_score,
            "n_distinct_clauses": len(clauses),
            "seed_among_candidates": seed in clauses,
            "n_trials": len(getattr(compiled, "trial_logs", {}) or {}),
            "trial_scores": sorted({round(float(c["score"]), 4) for c in cands
                                    if isinstance(c, dict) and c.get("score") is not None}),
            "best_trial_score": getattr(compiled, "score", None),


            "restates_instrument": restates_instrument(found, seed),
        }
        after = (clause_signature(compiled).output_fields["response"].json_schema_extra or {}).get("desc")
        if after != frozen_desc:
            raise SystemExit("the optimiser rewrote the OUTPUT FIELD description — the "
                             "`Answer: X / Why:` instrument is no longer frozen and nothing from "
                             "this run is comparable to the ladder. Do not use these results.")


    if out.exists() and not args.force and not args.dry_run:
        try:
            prev = json.loads(out.read_text())
        except (json.JSONDecodeError, OSError):
            prev = None
        if prev:
            keys = ("run", "model", "objective", "seed_prompt", "concepts",
                    "think_state", "prompt_model", "demos")
            now = {"run": args.run, "model": args.model, "objective": args.objective,
                   "seed_prompt": args.seed_prompt, "concepts": list(args.concept),
                   "think_state": args.think_state, "prompt_model": args.prompt_model,
                   "demos": args.demos}
            diff = {k: (prev.get(k), now[k]) for k in keys
                    if k in prev and prev.get(k) != now[k]}
            if diff:
                lines = "\n".join(f"    {k}: on disk {a!r}  ->  now {b!r}"
                                   for k, (a, b) in sorted(diff.items()))
                raise SystemExit(
                    f"{out} already holds a DIFFERENT experiment:\n{lines}\n"
                    f"Writing would destroy it. Use a new --round, or --force if you really mean "
                    f"to replace it.")
    _write({
        "git_commit": git_commit(), "run": args.run, "out_run": out_run, "model": args.model,
        "objective": args.objective, "seed_prompt": args.seed_prompt, "concepts": args.concept,
        "think_state": args.think_state, "dry_run": args.dry_run,


        "n_candidates_requested": args.candidates, "n_trials_requested": n_trials_req,
        "n_examples": len(examples), "n_train": len(trainset), "n_dev": len(valset),
        "max_tokens": args.max_tokens, "extra_body": body,
        "prompt_model": args.prompt_model, "round": args.round, "status": "complete",
        "probe_types": list(probe_types), "example_mix": dict(mix), "minibatch": args.minibatch,


        "demos": args.demos,
        "arms_requested_but_empty": missing,
        "dev_mix": dict(dev_mix),
        "seed_clause": seed, "found_clause": found,
        "seed_score": seed_score, "found_score": found_score,

        "seed_items": seed_items, "found_items": found_items, "search": search,
        "elapsed_s": round(time.time() - t0, 1),
    })

    print(f"\nseed  {seed_score:.4f}\nfound {found_score:.4f}   (Δ {found_score - seed_score:+.4f})")
    print(f"-> {out}")
    print("\nThe found clause is NOT a result yet: register it as a prompt_id and run it through the "
          "normal sweep + scorers, on `test`, before quoting any number from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
