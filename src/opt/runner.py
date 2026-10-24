from __future__ import annotations

import time
from pathlib import Path

from src.opt import result_path
from src.opt.adapter import VerbatimAdapter
from src.opt.candidates import describe_search
from src.opt.metrics import metric_for
from src.opt.preflight import preflight
from src.opt.preparation import prepare_examples
from src.opt.program import build_program, clause_of, clause_signature, instrument_is_frozen, seed_clause
from src.opt.results import check_destination, write_result
from src.opt.search import compile_search, default_trials
from src.opt.serving import extra_body
from src.paths import optimize_namespace
from src.provenance import git_commit


def run(args) -> int:


    if not args.prompt_model and args.think_state == "on":
        raise SystemExit(
            f"--think-state on with no --prompt-model: the target would propose its own clauses "
            f"while thinking, and its trace collides with DSPy's field markers — every proposal "
            f"fails to parse and the search dies without writing.\n"
            f"  Either pass PROMPT_MODEL=<a model served thinking-off> (D16, and the Llama "
            f"self-proposal result says to anyway: 7 of 19 candidates were meta-text),\n"
            f"  or run the target with --think-state off.")

    import dspy

    examples, trainset, valset, probe_types, mix, missing, dev_mix = prepare_examples(args)


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

    preflight(args, program, seed, examples, trainset, body)

    evaluate = dspy.Evaluate(devset=valset, metric=metric, num_threads=8,
                             display_progress=True, display_table=0)


    n_trials_req = args.trials if args.trials is not None else default_trials(args.candidates)
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

    check_destination(out, args)

    write_result(out, {"status": "seed-only — the search did not finish; this is NOT a result",
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
        compiled = compile_search(args, program, metric, trainset, valset, n_trials_req)
        found = clause_of(compiled)
        found_score = float(evaluate(compiled).score)
        found_items = per_item(compiled)
        search = describe_search(compiled, found, seed, seed_score)
        if not instrument_is_frozen(compiled, frozen_desc):
            raise SystemExit("the optimiser rewrote the OUTPUT FIELD description — the "
                             "`Answer: X / Why:` instrument is no longer frozen and nothing from "
                             "this run is comparable to the ladder. Do not use these results.")

    write_result(out, {
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
