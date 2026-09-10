#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.removal import Cell, Removal  # noqa: E402
from src.data.wmdp import NEAR_RETAIN  # noqa: E402


LADDERS: dict[str, list[tuple[str, str, float]]] = {
    "gemma-4 (think toggle)": [
        ("gemma-4-E2B-it", "E2B", 2.3),
        ("gemma-4-E4B-it", "E4B", 4.5),
        ("gemma-4-12B-it", "12B", 11.95),
        ("gemma-4-31B-it", "31B", 30.7),
    ],
    "Llama-3 (na, no reasoning mode)": [
        ("Meta-Llama-3-8B-Instruct", "8B", 8.0),
        ("Meta-Llama-3-70B-Instruct", "70B", 70.6),
    ],
}

METRICS = ("removable", "selectivity", "neighbour", "far", "forget_acc")


def _load(run: str):
    d = Path("results") / run / "analysis" / "test"
    fwd = d / "accuracy_delta.csv"
    if not fwd.exists():
        raise SystemExit(f"no {fwd} — run `python scripts/score_accuracy.py --run {run} "
                         f"--split test` first")
    ret = d / "side_effect.csv"
    return (list(csv.DictReader(open(fwd))),
            list(csv.DictReader(open(ret))) if ret.exists() else [])


def _cell(fwd_row, ret_rows, wave: str) -> Cell | None:

    if not fwd_row.get(f"{wave}_acc"):
        return None
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    near = set(NEAR_RETAIN.get(fwd_row["mmlu_subject"], ()))
    tiers = {}
    for name, want_near in (("neighbour", True), ("far", False)):
        sub = [r for r in ret_rows
               if (r["mmlu_subject"] in near) == want_near and r.get(f"{wave}_acc")]
        if sub:
            tiers[name] = Removal(mean([float(r["baseline_acc"]) for r in sub]),
                                  mean([float(r[f"{wave}_acc"]) for r in sub]))
    if "neighbour" not in tiers:
        return None
    return Cell(forget=Removal(float(fwd_row["baseline_acc"]), float(fwd_row[f"{wave}_acc"])),
                neighbour=tiers["neighbour"], far=tiers.get("far"))


def _value(cell: Cell | None, metric: str):
    if cell is None:
        return None
    if metric == "removable":
        return cell.forget.credited_pct
    if metric == "selectivity":
        return cell.selectivity_gap
    if metric == "neighbour":
        return cell.neighbour.drop_pp
    if metric == "far":
        return cell.far.drop_pp if cell.far is not None else None
    return cell.forget.suppressed * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="wmdp5.1")
    ap.add_argument("--metric", default="removable", choices=METRICS,
                    help='Metric.')
    ap.add_argument("--md", action="store_true", help='Md.')
    args = ap.parse_args()

    fwd, ret = _load(args.run)
    waves = sorted({c[:-4] for c in (fwd[0] if fwd else {}) if c.endswith("_acc")} - {"baseline"})
    if not waves:
        raise SystemExit(f"no scored waves in results/{args.run}/analysis/test/accuracy_delta.csv")

    lines: list[str] = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# {args.run} — the scale ladder · metric: {args.metric}")
    emit()
    emit("Rungs left-to-right by size. `—` = that rung has not generated this wave yet — NOT a zero.")
    emit()

    for ladder, rungs in LADDERS.items():
        present = [(m, lbl, b) for m, lbl, b in rungs
                   if any(r["model"] == m for r in fwd)]
        if not present:
            continue
        missing = [lbl for m, lbl, _ in rungs if not any(r["model"] == m for r in fwd)]
        emit(f"## {ladder}")
        if missing:
            emit(f"> ⚠️ not yet generated: {', '.join(missing)} — the trend below is incomplete.")
        emit()


        keys = sorted({(r["mmlu_subject"], r["think_state"]) for r in fwd
                       if any(r["model"] == m for m, _, _ in present)})
        for concept, think in keys:
            head = " | ".join(f"{lbl} ({b}B)" for _, lbl, b in present)
            emit(f"### {concept} · think={think}")
            emit()
            emit(f"| wave | {head} | Δ end-to-end |")
            emit("|---|" + "---|" * (len(present) + 1))
            for wave in waves:
                vals = []
                for model, _, _ in present:
                    row = next((r for r in fwd if r["model"] == model
                                and r["mmlu_subject"] == concept and r["think_state"] == think),
                               None)
                    if row is None:
                        vals.append(None)
                        continue
                    rows = [x for x in ret if x["model"] == model
                            and x["think_state"] == think and x["concept_id"] == concept]
                    vals.append(_value(_cell(row, rows, wave), args.metric))
                shown = " | ".join("—" if v is None else f"{v:.0f}" for v in vals)
                ends = [v for v in vals if v is not None]


                delta = (f"{ends[-1] - ends[0]:+.0f}"
                         if len(ends) >= 2 and vals[0] is not None and vals[-1] is not None
                         else "—")
                emit(f"| `{wave}` | {shown} | **{delta}** |")
            emit()

    emit("Read the DIRECTION and the MAGNITUDE, not a correlation: four rungs (two for Llama) "
         "cannot support one. If the sign disagrees across waves, that disagreement is the result.")

    if args.md:
        out = Path("results") / args.run / "presentation" / "scale_ladder.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
