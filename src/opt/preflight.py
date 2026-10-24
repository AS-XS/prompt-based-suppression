from __future__ import annotations

from src.opt.program import build_program
from src.opt.serving import check_think_state


def preflight(args, program, seed, examples, trainset, body) -> None:
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
