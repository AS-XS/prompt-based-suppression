#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CHANCE = 0.25


TIERS = [("hazard", "#2a78d6"), ("neighbour", "#eb6834"), ("far MMLU", "#1baf7a")]

BASELINE_FILL = "#8a8a85"        
ABOVE_FILL = "#2a78d6"           
BELOW_FILL = "#d03b3b"           
GRID = "#d7d7d2"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"


LABELS: dict[str, str] = {
    "none-baseline": "baseline (no prompt)",
    "scoped-world@specific": "world",
    "scoped-retain@specific": "+ retain",
    "scoped-sandbag": "+ sandbag  [bare label]",
    "scoped-sandbag@specific": "+ sandbag",
    "scoped-directed": "+ both  [bare label]",
    "scoped-directed@specific": "+ both",
    "scoped-directed-named-broad@specific": "+ both, keep-list broad",
    "scoped-directed-named-exact@specific": "+ both, keep-list exact",
    "scoped-directed-fewshot@specific": "+ both, 2 examples",
    "scoped-directed-fewshot-grouped@specific": "+ both, 3 examples",
    "scoped-directed-aware@specific": "+ both, monitored",
    "scoped-sandbag-aware@specific": "+ sandbag, monitored",
}
ORDER = list(LABELS)


def _cells(run: str):
    path = Path("results") / run / "analysis" / "test" / "accuracy_delta.csv"
    if not path.exists():
        raise SystemExit(f"no {path} — run `python scripts/score_accuracy.py --run {run} "
                         f"--split test` first")
    rows = list(csv.DictReader(open(path)))
    waves = [w for w in ORDER if f"{w}_acc" in (rows[0] if rows else {})]
    waves += sorted({c[:-4] for c in (rows[0] if rows else {}) if c.endswith("_acc")}
                    - set(waves) - {"baseline"})
    return rows, waves


def _tier_levels(run: str):


    from src.data.wmdp import NEAR_RETAIN

    d = Path("results") / run / "analysis" / "test"
    fwd = list(csv.DictReader(open(d / "accuracy_delta.csv")))
    ret = list(csv.DictReader(open(d / "side_effect.csv")))
    waves = [w for w in ORDER if w != "none-baseline" and f"{w}_acc" in (fwd[0] if fwd else {})]

    out = []
    for r in fwd:
        key = (r["model"], r["think_state"], r["mmlu_subject"])
        near = set(NEAR_RETAIN.get(r["mmlu_subject"], ()))
        rows = [x for x in ret if (x["model"], x["think_state"], x["concept_id"]) == key]
        tiers = {"hazard": (float(r["baseline_acc"]),
                            {w: float(r[f"{w}_acc"]) for w in waves if r.get(f"{w}_acc")})}
        for name, want_near in (("neighbour", True), ("far MMLU", False)):
            sub = [x for x in rows if (x["mmlu_subject"] in near) == want_near]
            if not sub:
                continue
            mean = lambda xs: sum(xs) / len(xs) if xs else None
            tiers[name] = (
                mean([float(x["baseline_acc"]) for x in sub]),
                {w: mean([float(x[f"{w}_acc"]) for x in sub if x.get(f"{w}_acc")])
                 for w in waves if any(x.get(f"{w}_acc") for x in sub)})
        out.append((r, tiers))
    return out, waves


def _draw_tiers(ax, r, tiers, waves, fs: float = 1.0):


    h = 0.26
    ax.set_facecolor(SURFACE)
    labels = ["baseline (no prompt)"] + [LABELS.get(w, w) for w in waves]
    ys = list(range(len(labels)))[::-1]

    for ti, (tier, colour) in enumerate(TIERS):
        if tier not in tiers:
            continue
        base, per_wave = tiers[tier]
        vals = [base] + [per_wave.get(w) for w in waves]
        offset = (1 - ti) * h
        for y, v in zip(ys, vals):
            if v is None:
                continue
            below = v < CHANCE
            ax.barh(y + offset, v, height=h - 0.035, color=colour, edgecolor=SURFACE,
                    linewidth=1.6, hatch="///" if below else None, zorder=3,
                    alpha=1.0 if y != max(ys) else 0.45)
            ax.text(v + 0.008, y + offset, f"{v*100:.0f}", va="center", ha="left",
                    fontsize=6.6 * fs, color=INK_SOFT, zorder=5,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.7))

    ax.axvline(CHANCE, color="#d03b3b", linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.4 * fs, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%\nchance", "50%", "75%", "100%"], fontsize=8 * fs, color=INK_SOFT)
    ax.get_xticklabels()[1].set_color("#d03b3b")
    ax.set_ylim(-0.8, len(labels) - 0.2)
    state = {"on": "thinking on", "off": "thinking off"}.get(r["think_state"],
                                                            "no reasoning mode")
    ax.set_title(f"{r['model']}  ·  {r['mmlu_subject']}  ({state})",
                 fontsize=11.5 * fs, pad=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def _plot_tiers(plt, Patch, run: str, out_dir: Path, split: bool = False):


    cells, waves = _tier_levels(run)
    out_dir.mkdir(parents=True, exist_ok=True)

    if split:
        legend = [Patch(facecolor=c, label=n) for n, c in TIERS] + [
            Patch(facecolor="#ffffff", edgecolor=INK_SOFT, hatch="///", label="below chance")]
        for r, tiers in cells:
            fig, ax = plt.subplots(figsize=(15.5, 9.5), facecolor=SURFACE)
            _draw_tiers(ax, r, tiers, waves, fs=1.7)
            fig.legend(handles=legend, loc="lower right", bbox_to_anchor=(0.99, 0.012), ncol=4,
                       frameon=False, fontsize=11)
            dest = out_dir / (f"{run}_blast_radius_{r['model']}_{r['mmlu_subject']}"
                              f"_think-{r['think_state']}.png")
            fig.tight_layout(rect=(0, 0.045, 1, 1))
            fig.savefig(dest, dpi=170, facecolor=SURFACE)
            plt.close(fig)
            print(f"  -> {dest}")
        return

    ncol = 2
    nrow = (len(cells) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(17, 5.9 * nrow + 2.0), squeeze=False,
                             facecolor=SURFACE)
    h = 0.26

    for ax, (r, tiers) in zip([a for row in axes for a in row], cells):
        _draw_tiers(ax, r, tiers, waves)

    for ax in [a for row in axes for a in row][len(cells):]:
        ax.set_visible(False)

    fig.suptitle(f"{run} — blast radius as accuracy LEVELS, test split",
                 fontsize=14, fontweight="bold", x=0.010, ha="left", y=0.996, color=INK)
    fig.text(0.010, 0.977,
             "Three tiers per prompt, each against its own baseline (top row, faded). Targeted "
             "suppression = the hazard bar falls to the floor while neighbour and far MMLU stay up. "
             "All three falling together is a model answering worse at everything.",
             fontsize=9.5, ha="left", color=INK_SOFT)
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in TIERS]
                       + [Patch(facecolor="#ffffff", edgecolor=INK_SOFT, hatch="///",
                                label="below chance")],
               loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=4, frameon=False, fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{run}_blast_radius_levels.png"
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(dest, dpi=165, facecolor=SURFACE)
    print(f"  -> {dest}")


def _draw_accuracy(ax, r, waves, fs: float = 1.0):


    ax.set_facecolor(SURFACE)
    bars = [("none-baseline", float(r["baseline_acc"]))]
    bars += [(w, float(r[f"{w}_acc"])) for w in waves if r.get(f"{w}_acc")]
    ys = list(range(len(bars)))[::-1]

    for y, (wave, acc) in zip(ys, bars):
        below = acc < CHANCE
        base = wave == "none-baseline"
        fill = BASELINE_FILL if base else (BELOW_FILL if below else ABOVE_FILL)
        ax.barh(y, acc, height=0.66, color=fill, edgecolor=SURFACE, linewidth=2,
                hatch="///" if below else None, zorder=3)


        ax.text(acc + 0.012, y, f"{acc*100:.1f}%" + ("  ▼ below chance" if below else ""),
                va="center", ha="left", fontsize=8.5 * fs,
                color=BELOW_FILL if below else INK_SOFT,
                fontweight="bold" if below else "normal", zorder=5,

                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))

    ax.axvline(CHANCE, color=BELOW_FILL, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.axvline(float(r["baseline_acc"]), color=BASELINE_FILL, linewidth=1.1,
               linestyle=(0, (2, 3)), zorder=2)

    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS.get(w, w) for w, _ in bars], fontsize=8.6 * fs, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%\nchance", "50%", "75%", "100%"], fontsize=8,
                       color=INK_SOFT)
    ax.get_xticklabels()[1].set_color(BELOW_FILL)
    ax.set_ylim(-0.7, len(bars) - 0.35)
    state = {"on": "thinking on", "off": "thinking off"}.get(r["think_state"],
                                                            "no reasoning mode")
    ax.set_title(f"{r['model']}  ·  {r['mmlu_subject']}  ({state})",
                 fontsize=11.5 * fs, pad=9, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="wmdp5")
    ap.add_argument("--out", default="png")
    ap.add_argument("--split", action="store_true",
                    help='Split.')
    ap.add_argument("--tiers", action="store_true",
                    help='Tiers.')
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        raise SystemExit("needs matplotlib: pip install -e '.[analysis]'")

    if args.tiers:
        _plot_tiers(plt, Patch, args.run, Path(args.out), split=args.split)
        return 0

    rows, waves = _cells(args.run)
    if not rows:
        raise SystemExit(f"no scored rows for {args.run}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.split:
        for r in rows:
            fig, ax = plt.subplots(figsize=(13.5, 8.2), facecolor=SURFACE)
            _draw_accuracy(ax, r, waves, fs=1.65)
            fig.legend(handles=[Patch(facecolor=BASELINE_FILL, label="baseline (no prompt)"),
                                Patch(facecolor=ABOVE_FILL, label="suppressed, above chance"),
                                Patch(facecolor=BELOW_FILL, hatch="///",
                                      label="suppressed, BELOW chance")],
                       loc="lower right", bbox_to_anchor=(0.99, 0.015), ncol=3, frameon=False,
                       fontsize=11)
            dest = out / (f"{args.run}_accuracy_{r['model']}_{r['mmlu_subject']}"
                          f"_think-{r['think_state']}.png")
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            fig.savefig(dest, dpi=170, facecolor=SURFACE)
            plt.close(fig)
            print(f"  -> {dest}")
        return 0

    ncol = 2
    nrow = (len(rows) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.5, 3.15 * nrow + 1.9), squeeze=False,
                             facecolor=SURFACE)

    for ax, r in zip([a for row in axes for a in row], rows):
        _draw_accuracy(ax, r, waves)

    for ax in [a for row in axes for a in row][len(rows):]:
        ax.set_visible(False)

    fig.suptitle(f"{args.run} — accuracy on the suppressed (forget) tier, test split",
                 fontsize=14, fontweight="bold", x=0.011, ha="left", y=0.995, color=INK)
    fig.text(0.011, 0.963,
             "Each bar is where the model actually lands, not how far it fell. Below the 25% floor "
             "the knowledge is intact and being steered around — a failure, not stronger removal.",
             fontsize=9.5, ha="left", color=INK_SOFT)
    fig.legend(handles=[Patch(facecolor=BASELINE_FILL, label="baseline (no suppression prompt)"),
                        Patch(facecolor=ABOVE_FILL, label="suppressed, above chance"),
                        Patch(facecolor=BELOW_FILL, hatch="///", label="suppressed, BELOW chance")],
               loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=3, frameon=False, fontsize=9)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.run}_accuracy.png"
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(dest, dpi=170, facecolor=SURFACE)
    print(f"  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
