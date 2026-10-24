from __future__ import annotations

from src.opt.serving import extra_body


def default_trials(n_candidates: int, n_predictors: int = 1) -> int:


    import math

    return int(max(2 * n_predictors * math.log2(max(n_candidates, 2)), 1.5 * n_candidates))


def compile_search(args, program, metric, trainset, valset, n_trials_req):
    import dspy
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
    return compiled
