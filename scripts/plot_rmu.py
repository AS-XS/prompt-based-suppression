#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.wmdp import NEAR_RETAIN  # noqa: E402
from src.paths import analysis_dir, generations_dir  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")             
CORRECT, WRONG, NONE = "#4a3aa7", "#9085e9", "#e87ba4"             
CHANCE = 25.0


def _strict(rows, model, *, think=None, subjects=None, probe=None) -> tuple[float, int, int, int]:


    sel = [r for r in rows if r["model"] == model
           and (think is None or r.get("think_state") == think)
           and (subjects is None or r["mmlu_subject"] in subjects)
           and (probe is None or r["probe_type"] == probe)]
    n = sum(int(r["n"]) for r in sel)
    p = sum(int(r["n_parsed"]) for r in sel)
    c = sum(int(r["n_correct"]) for r in sel)
    return (100.0 * c / n if n else float("nan")), n, p, c


def _cap_share(run: str, model: str, target: str, think=None) -> tuple[int, int]:


    from src.analysis.accuracy import extract_letter_tiered
    from src.analysis.surfaces import answer_of

    capped = unparsed = 0
    for f in glob.glob(str(generations_dir(run) / f"{model}__*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("probe_type") != "forget" or r.get("concept_id") != target:
                continue
            ts = r.get("think_state")
            if think is not None and ts != think:
                continue
            if extract_letter_tiered(answer_of(r.get("raw_output"), expect_trace=ts == "on"))[0] is None:
                unparsed += 1
                capped += r.get("finish_reason") == "length"
    return capped, unparsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline-model", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True, help='Checkpoints.')
    ap.add_argument("--labels", nargs="+", default=None, help='Labels.')
    ap.add_argument("--target", default="wmdp-cyber", help='Target.')
    ap.add_argument("--control", default="wmdp-bio", help='Control.')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if len(args.checkpoints) > 3:
        raise SystemExit("at most 3 checkpoints: four series is where the validated palette's "
                         "adjacent-pair guarantee ends for grouped bars")
    models = [args.baseline_model, *args.checkpoints]
    labels = ["unedited", *(args.labels or args.checkpoints)]
    if len(labels) != len(models):
        raise SystemExit("--labels must match --checkpoints one-for-one")

    path = analysis_dir(args.run) / "test" / "accuracy.csv"
    if not path.is_file():
        raise SystemExit(f"no {path} — run score_accuracy.py --run {args.run} --split test first")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    missing = [m for m in models if not any(r["model"] == m for r in rows)]
    if missing:
        raise SystemExit(f"not scored in {args.run}: {missing}")

    t, c = args.target.removeprefix("wmdp-"), args.control.removeprefix("wmdp-")
    tiers = [(f"forget\n{t} (target)", dict(subjects={args.target})),
             (f"forget\n{c} (control)", dict(subjects={args.control})),
             (f"neighbour\n{t}", dict(subjects=set(NEAR_RETAIN[args.target]))),
             (f"neighbour\n{c}", dict(subjects=set(NEAR_RETAIN[args.control]))),
             ("far MMLU", dict(probe="utility"))]


    states = sorted({r.get("think_state") for r in rows if r["model"] == args.baseline_model}, key=str)
    if args.out and len(states) > 1:
        raise SystemExit(f"--out names ONE file but {args.run} has think states {states}; leave "
                         f"--out off and one figure is written per state")
    for think in states:
        acc = {m: [_strict(rows, m, think=think, **kw) for _, kw in tiers] for m in models}
        suffix = "" if len(states) == 1 else f"_think-{think}"
        out = Path(args.out) if args.out else Path("png") / f"{args.run}_unlearning{suffix}.png"
        shown = think if len(states) > 1 else None
        _print_table(acc, tiers, models, labels, shown)
        _draw(args, acc, tiers, models, labels, shown, out, t)
    return 0


def _print_table(acc, tiers, models, labels, think) -> None:


    from src.analysis.removal import Removal

    base = models[0]
    print(f"\n### think {think}" if think else "")
    print("| tier | n | " + " | ".join(labels) + " |")
    print("|---|---:|" + "---|" * len(labels))
    for k, (name, _) in enumerate(tiers):
        b = acc[base][k][0]
        cells = [f"{b:.1f}%"]
        for m in models[1:]:
            a = acc[m][k][0]
            r = Removal(baseline=b / 100, suppressed=a / 100)
            change = -r.drop_pp if abs(r.drop_pp) >= 0.05 else 0.0
            cell = f"{a:.1f}% · {change:+.1f}"
            if name.startswith("forget") and r.pct_removable is not None:
                cell += f" · {r.credited_pct:.0f}%" + (" ⚠" if r.below_chance else "")
            cells.append(cell)
        print(f"| {name.replace(chr(10), ' ')} | {acc[base][k][1]:,} | " + " | ".join(cells) + " |")


def _draw(args, acc, tiers, models, labels, think, out: Path, t: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10.5, 8.6), facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [1.55, 1], "hspace": 0.62})


    ax.set_facecolor(SURFACE)
    width = 0.8 / len(models)
    moving = {0, 2}
    for i, (m, lab) in enumerate(zip(models, labels)):
        xs = [k + (i - (len(models) - 1) / 2) * width for k in range(len(tiers))]
        ys = [acc[m][k][0] for k in range(len(tiers))]
        ax.bar(xs, ys, width * 0.94, color=SERIES[i], label=lab, edgecolor=SURFACE, linewidth=1.2,
               zorder=3)
        for k in moving:
            ax.text(xs[k], ys[k] + 1.2, f"{ys[k]:.0f}", ha="center", va="bottom", fontsize=8,
                    color=INK_2, zorder=4)
    ax.axhline(CHANCE, color=MUTED, linewidth=1.0, zorder=2)


    ax.annotate("chance\n(25%)", xy=(1.006, CHANCE), xycoords=("axes fraction", "data"),
                ha="left", va="center", fontsize=8, color=MUTED, annotation_clip=False)
    ax.set_xticks(range(len(tiers)), [name for name, _ in tiers])
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_ylabel("strict accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(models), frameon=False,
              fontsize=9, handlelength=1.2, columnspacing=1.6)
    ax.set_title("A   Accuracy by tier, against the unedited model", loc="left", fontsize=11,
                 color=INK, pad=30)


    bx.set_facecolor(SURFACE)
    for i, m in enumerate(models):
        _, n, p, cor = acc[m][0]
        parts = [(100 * cor / n, CORRECT, "white"), (100 * (p - cor) / n, WRONG, INK),
                 (100 * (n - p) / n, NONE, INK)]
        y, left = len(models) - 1 - i, 0.0
        for share, colour, ink in parts:
            bx.barh(y, share, left=left, height=0.62, color=colour, edgecolor=SURFACE, linewidth=1.2)
            if share >= 6:
                bx.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center", fontsize=8.5,
                        color=ink)
            left += share
    bx.set_yticks(range(len(models)), list(reversed(labels)))
    bx.set_xlim(0, 100)
    bx.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
    for side in ("top", "right", "left"):
        bx.spines[side].set_visible(False)
    bx.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in (CORRECT, WRONG, NONE)]
    bx.legend(handles, ["correct", "answered, wrong", "no answer"], loc="upper center",
              bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=9, handlelength=1.2)
    bx.set_title(f"B   What the {t} answers were", loc="left",
                 fontsize=11, color=INK, pad=26)


    base_name = args.baseline_model

    state = f"  (think {think})" if think else ""
    fig.suptitle(f"RMU on {base_name}, trained to remove {args.target} — against the unedited "
                 f"model{state}",
                 fontsize=12.5, color=INK, x=0.07, ha="left", y=0.985)
    notes = []
    for m, lab in zip(args.checkpoints, labels[1:]):
        capped, unparsed = _cap_share(args.run, m, args.target, think)
        if unparsed:
            notes.append(f"{lab}: {100 * capped / unparsed:.0f}%")
    n_by_tier = " · ".join(f"{name.replace(chr(10), ' ')} {acc[base_name][k][1]:,}"
                           for k, (name, _) in enumerate(tiers))
    footer = (f"Run {args.run}: strict accuracy, a non-answer counts as wrong; forget tiers are the test "
              f"split. n: {n_by_tier}.\n"
              f"Share of unparsed {t} responses reaching the token limit (all forget items): "
              + ("; ".join(notes) if notes else "no unparsed responses") + ".")
    fig.text(0.07, 0.012, footer, fontsize=7.8, color=INK_2, va="bottom", ha="left")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
