from __future__ import annotations

import json
from pathlib import Path


def write_result(out: Path, payload: dict) -> None:

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)


def check_destination(out: Path, args) -> None:
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
