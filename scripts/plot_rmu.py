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
CAPPED = "#eda100"                           
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
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help='Checkpoints.')
    ap.add_argument("--labels", nargs="+", default=None, help='Labels.')
    ap.add_argument("--target", default="wmdp-cyber", help='Target.')
    ap.add_argument("--control", default="wmdp-bio", help='Control.')
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None,
                    help='Name.')
    ap.add_argument("--table", action="store_true",
                    help='Table.')
    ap.add_argument("--answers", action="store_true",
                    help='Answers.')
    args = ap.parse_args()

    grid = args.table or args.answers
    if args.table and args.answers:
        raise SystemExit("--table and --answers are two views: run the script once for each")
    if args.checkpoints and len(args.checkpoints) > 3 and not grid:
        raise SystemExit("at most 3 checkpoints in the figure: four series is where the validated "
                         "palette's adjacent-pair guarantee ends for grouped bars. Use --table for "
                         "the whole grid.")
    if not args.checkpoints and not grid:
        raise SystemExit("--checkpoints is required for the figure (at most 3); --table and "
                         "--answers take every checkpoint in the run without it")

    path = analysis_dir(args.run) / "test" / "accuracy.csv"
    if not path.is_file():
        raise SystemExit(f"no {path} — run score_accuracy.py --run {args.run} --split test first")
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not args.checkpoints:
        args.checkpoints = sorted({r["model"] for r in rows} - {args.baseline_model})
    models = [args.baseline_model, *args.checkpoints]
    labels = ["unedited", *(args.labels or [_row_label(m) for m in args.checkpoints])]
    if len(labels) != len(models):
        raise SystemExit("--labels must match --checkpoints one-for-one")
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
    if args.table:
        stem = args.name or f"{args.run}_table"
        _draw_table(rows, args, models, labels, states, Path("png") / f"{stem}.png")
        return 0
    if args.answers:
        stem = args.name or f"{args.run}_answers"
        short = ["unedited", *(args.labels or [_short_label(m) for m in args.checkpoints])]
        _draw_answers(args, models, short, states, Path("png") / f"{stem}.png")
        return 0
    if args.out and len(states) > 1:
        raise SystemExit(f"--out names ONE file but {args.run} has think states {states}; leave "
                         f"--out off and one figure is written per state")
    for think in states:
        acc = {m: [_strict(rows, m, think=think, **kw) for _, kw in tiers] for m in models}
        suffix = "" if len(states) == 1 else f"_think-{think}"
        stem = args.name or f"{args.run}_unlearning"
        out = Path(args.out) if args.out else Path("png") / f"{stem}{suffix}.png"
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
                cell += f" · {_pct(r.credited_pct)}" + (" ⚠" if r.below_chance else "")
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
    fig.suptitle(f"{_method_of(args.checkpoints)} on {base_name}, trained to remove {args.target} — "
                 f"against the unedited model{state}",
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


def _target_of(name: str) -> str | None:


    tokens = set(name.split("-"))
    hits = [c for c in ("bio", "cyber") if c in tokens]
    return f"wmdp-{hits[0]}" if len(hits) == 1 else None


def _pct(value: float) -> str:

    return f"{0 if abs(value) < 0.5 else value:.0f}%"


def _method_of(checkpoints) -> str:


    methods = {name.split("-", 1)[0] for name in checkpoints}
    return {frozenset({"rmu"}): "RMU", frozenset({"elm"}): "ELM"}.get(frozenset(methods), "Unlearning")


def _short_label(name: str) -> str:


    tokens = name.split("-")
    for i, tok in enumerate(tokens):
        if tok in ("bio", "cyber"):
            return "-".join(tokens[i + 1:]) or name
    return name


def _row_label(name: str) -> str:

    target, short = _target_of(name), _short_label(name)
    return f"{target.removeprefix('wmdp-')} · {short}" if target and short != name else name


def _answer_split(run: str, tables: dict, model: str, concept: str, think) -> dict[str, int]:


    n = parsed = correct = 0
    for rows in tables.values():
        _, n_s, p_s, c_s = _strict(rows, model, think=think, subjects={concept})
        n, parsed, correct = n + n_s, parsed + p_s, correct + c_s
    capped, unparsed = _cap_share(run, model, concept, think)
    if unparsed != n - parsed:
        raise SystemExit(f"{run} · {model} · {concept} · think {think}: the generations hold "
                         f"{unparsed} unanswered items, the scored tables {n - parsed}. Rescore "
                         f"(score_accuracy.py, both splits) before plotting.")
    return {"n": n, "correct": correct, "wrong": parsed - correct, "stopped": unparsed - capped,
            "capped": capped}


def _draw_answers(args, models, labels, states, out: Path) -> None:


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = models[0]
    unassigned = [m for m in models[1:] if _target_of(m) is None]
    if unassigned:
        raise SystemExit(f"cannot tell which domain these were trained to remove: {unassigned}. "
                         f"This view groups rows by target, read off the name (…-bio-… / …-cyber-…).")
    groups = []
    for concept in ("wmdp-cyber", "wmdp-bio"):
        members = [(m, lab) for m, lab in zip(models[1:], labels[1:]) if _target_of(m) == concept]
        if members:
            groups.append((concept, [(base, "unedited"), *members]))
    tables = {s: list(csv.DictReader(open(analysis_dir(args.run) / s / "accuracy.csv",
                                          encoding="utf-8"))) for s in ("val", "test")}
    counts = {(m, c, t): _answer_split(args.run, tables, m, c, t)
              for c, members in groups for m, _ in members for t in states}

    parts = [("correct", CORRECT, "white"), ("wrong", WRONG, INK), ("stopped", NONE, INK),
             ("capped", CAPPED, INK)]
    print(f"\n| target | checkpoint | think | n | correct | answered, wrong | no answer, stopped "
          f"| no answer, hit token cap |\n|---|---|---|---:|---:|---:|---:|---:|")
    for c, members in groups:
        for m, lab in members:
            for t in states:
                k = counts[(m, c, t)]
                print(f"| {c.removeprefix('wmdp-')} | {lab} | {t} | {k['n']:,} | "
                      + " | ".join(f"{100 * k[p] / k['n']:.1f}%" for p, _, _ in parts) + " |")

    ys, heads, y = [], [], 0.0
    for c, members in groups:
        heads.append((y, c, counts[(members[0][0], c, states[0])]["n"]))
        y += 0.9
        for m, lab in members:
            ys.append((y, c, m, lab))
            y += 1.0
        y += 0.5


    titled = any(t not in (None, "na") for t in states)          
    top_in, bottom_in, left_in, col_in = (1.25 if titled else 1.05), 0.75, 1.45, 5.2
    gap_in = 0.6
    fig_w = left_in + col_in * len(states) + gap_in * (len(states) - 1) + 0.3
    fig_h = top_in + 0.34 * y + bottom_in
    fig, axes = plt.subplots(1, len(states), figsize=(fig_w, fig_h), facecolor=SURFACE, sharey=True,
                             squeeze=False)
    fig.subplots_adjust(left=left_in / fig_w, right=1 - 0.3 / fig_w, top=1 - top_in / fig_h,
                        bottom=bottom_in / fig_h, wspace=gap_in / col_in)
    for ax, think in zip(axes[0], states):
        ax.set_facecolor(SURFACE)
        for yy, c, m, _ in ys:
            k, left = counts[(m, c, think)], 0.0
            for part, colour, ink in parts:
                share = 100 * k[part] / k["n"]
                ax.barh(yy, share, left=left, height=0.72, color=colour, edgecolor=SURFACE,
                        linewidth=1.6)
                if share >= 7:
                    ax.text(left + share / 2, yy, f"{share:.0f}%", ha="center", va="center",
                            fontsize=8.5, color=ink)
                left += share
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.set_ylim(y - 0.4, -0.55)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if think not in (None, "na"):
            ax.set_title(f"think {think}", loc="left", fontsize=11, color=INK, pad=6)


        for yy, c, n in heads:
            ax.text(0, yy, f"trained to remove {c.removeprefix('wmdp-')}  ({n:,} items)", ha="left",
                    va="center", fontsize=9.5, color=INK, weight="bold")
    axes[0][0].set_yticks([yy for yy, *_ in ys], [lab for *_, lab in ys])
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for _, colour, _ in parts]
    fig.legend(handles, ["correct", "answered, wrong", "no answer, stopped",
                         "no answer, hit token cap"], loc="upper left", ncol=4, frameon=False,
               fontsize=9.5, handlelength=1.2, bbox_to_anchor=(0.005, 1 - 0.66 / fig_h))

    fig.suptitle(f"{_method_of(models[1:])} on {base} — run {args.run}", fontsize=12.5, color=INK,
                 x=0.01, ha="left", y=1 - 0.1 / fig_h)
    fig.text(0.01, 1 - 0.4 / fig_h, "What each checkpoint answered on the domain it was trained to "
             "remove", fontsize=10, color=INK_2, ha="left", va="top")
    fig.text(0.01, 0.04 / fig_h, "All forget items of that domain, validation + test. No answer = "
             "no parseable letter, by the scorers' rule.\nHit token cap = the generation stopped at "
             "the token limit; stopped = it ended on its own.", fontsize=8, color=INK_2, ha="left",
             va="bottom")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def _draw_table(rows, args, models, labels, states, out: Path) -> None:


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.analysis.removal import Removal

    cols = [("forget\ncyber", "wmdp-cyber", dict(subjects={"wmdp-cyber"})),
            ("forget\nbio", "wmdp-bio", dict(subjects={"wmdp-bio"})),
            ("neighbour\ncyber", None, dict(subjects=set(NEAR_RETAIN["wmdp-cyber"]))),
            ("neighbour\nbio", None, dict(subjects=set(NEAR_RETAIN["wmdp-bio"]))),
            ("far\nMMLU", None, dict(probe="utility"))]
    header = ["", *[c[0] for c in cols], "answered\ncyber · bio"]
    base = models[0]
    tint, head, base_bg = "#ece9fb", "#efeee9", "#f6f5f1"


    row_h, pad = 0.33, 0.55
    tbl_h = row_h * (len(models) + 1.4)
    fig_h = 2 * pad + len(states) * tbl_h + (len(states) - 1) * 0.55
    fig = plt.figure(figsize=(14.5, fig_h), facecolor=SURFACE)
    gs = fig.add_gridspec(len(states), 1, hspace=0.55 / tbl_h, top=1 - pad / fig_h,
                          bottom=pad / fig_h, left=0.02, right=0.98)
    axes = [fig.add_subplot(gs[i]) for i in range(len(states))]
    for ax, think in zip(axes, states):
        ax.axis("off")
        b = {k: _strict(rows, base, think=think, **kw) for k, (_, _, kw) in enumerate(cols)}
        text, bold = [], []
        for m, lab in zip(models, labels):
            target = None if m == base else _target_of(m)
            line, marks = [lab], []
            for k, (_, concept, kw) in enumerate(cols):
                a = _strict(rows, m, think=think, **kw)[0]
                if m == base:
                    line.append(f"{a:.1f}")
                    continue
                r = Removal(baseline=b[k][0] / 100, suppressed=a / 100)
                change = -r.drop_pp if abs(r.drop_pp) >= 0.05 else 0.0
                cell = f"{a:.1f}  ({change:+.1f})"
                if concept is not None and concept == target and r.pct_removable is not None:
                    cell += f"   {_pct(r.credited_pct)}" + (" ⚠" if r.below_chance else "")
                    marks.append(k + 1)
                line.append(cell)
            pc = _strict(rows, m, think=think, subjects={"wmdp-cyber"})
            pb = _strict(rows, m, think=think, subjects={"wmdp-bio"})
            line.append(f"{100 * pc[2] / pc[1]:.0f} · {100 * pb[2] / pb[1]:.0f}")
            text.append(line)
            bold.append(marks)
        tbl = ax.table(cellText=text, colLabels=header, cellLoc="center", bbox=[0, 0, 1, 1],
                       colWidths=[0.25, 0.13, 0.13, 0.12, 0.12, 0.1, 0.12])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        for (i, j), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.8)
            if i == 0:
                cell.set_facecolor(head)
                cell.set_text_props(weight="bold", color=INK, fontsize=9)
            elif i == 1:
                cell.set_facecolor(base_bg)
                cell.set_text_props(weight="bold", color=INK)
            else:
                cell.set_facecolor(tint if j in bold[i - 1] else SURFACE)
                cell.set_text_props(color=INK, weight="bold" if j in bold[i - 1] else "normal")
            if j == 0 and i > 0:
                cell.set_text_props(ha="left")
                cell._loc = "left"
        ax.set_title(f"think {think}" if think not in (None, "na") else "", loc="left", fontsize=11,
                     color=INK, pad=5)

    fig.suptitle(f"{_method_of(models[1:])} on {args.baseline_model} — every checkpoint against the "
                 f"unedited model (run {args.run})", fontsize=12.5, color=INK, x=0.02, ha="left",
                 y=1 - 0.12 / fig_h)
    footer = ("Strict accuracy (%), a non-answer counts as wrong; forget tiers are the test split. "
              "(±) = change vs the unedited row, pp.\nHighlighted = the concept that checkpoint was "
              "trained to remove, with its share of removable knowledge (capped at 100%; ⚠ = below "
              "chance). Answered = share of forget items with a parseable answer.")
    fig.text(0.02, 0.1 / fig_h, footer, fontsize=8, color=INK_2, va="bottom", ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
