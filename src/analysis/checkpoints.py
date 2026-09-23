from __future__ import annotations

from typing import Iterable


def refuse_cross_family(models: Iterable[str], reference: str) -> None:

    from src.gen.thinkstate import inherited_from

    bad = []
    for m in sorted(set(models) - {reference}):
        base = inherited_from(m)
        if base != reference:
            bad.append((m, base))
    if bad:
        lines = "\n".join(
            f"    {m}: " + ("not an edited checkpoint" if b is None else f"edited from {b!r}")
            for m, b in bad)
        raise SystemExit(
            f"--baseline-model {reference!r} would be paired against models that are not its own "
            f"checkpoints:\n{lines}\n"
            f"A model contrast pairs every model in the run against ONE reference, item by item. "
            f"Keep each base model and its checkpoints in their own run (rmu2 = gemma-4-E2B, "
            f"rmu3 = Llama-3-8B), or the drop is computed against the wrong model — silently.")
