#!/usr/bin/env python


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.report import (  # noqa: E402
    CSS, html_heatmap, html_table, load, prompt_drops, terminal_heatmap, terminal_matrix,
)
from src.paths import analysis_dir, stale_analysis  # noqa: E402


def _fmt(rows: list[dict], cols: list[str] | None = None, limit: int = 40) -> str:

    if not rows:
        return "  (not scored yet)\n"
    cols = cols or list(rows[0])
    w = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    out = ["  " + "  ".join(c.ljust(w[c]) for c in cols),
           "  " + "  ".join("-" * w[c] for c in cols)]
    for r in rows[:limit]:
        out.append("  " + "  ".join(str(r.get(c, "")).ljust(w[c]) for c in cols))
    if len(rows) > limit:
        out.append(f"  … {len(rows)-limit} more rows")
    return "\n".join(out) + "\n"


def _by_model_state(rows: list[dict]):


    for model, ts in sorted({(r["model"], r["think_state"]) for r in rows}):
        yield model, ts, [r for r in rows if r["model"] == model and r["think_state"] == ts]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp1", help='Run.')
    ap.add_argument("--html", action="store_true", help='Html.')
    ap.add_argument("--no-colour", action="store_true", help='No colour.')
    ap.add_argument("--save", metavar="DIR", default=None,
                    help='Save.')
    args = ap.parse_args()


    if getattr(args, "run", None):
        _old = stale_analysis(args.run)
        if _old:
            print(f"⚠️  {len(_old)} analysis CSV(s) are older than the generations — this report "
                  f"would describe a stale scoring.\n    Refresh: python scripts/score_accuracy.py "
                  f"--run {args.run}\n", file=sys.stderr)

    d = analysis_dir(args.run)
    acc = load(d / "accuracy.csv")
    delta = load(d / "accuracy_delta.csv")
    side = load(d / "side_effect.csv")
    matrix = load(d / "subject_matrix.csv")
    from src.analysis.summary import LEGEND, cross_model_rows
    summary = cross_model_rows(d)
    recall = load(d / "recall.csv")


    rating = load(d / "rating_distribution.csv")
    modes = load(d / "mode_distribution.csv")
    cond = load(d / "conditional_stats.csv")

    if not any([acc, delta, side, matrix, recall, rating, modes, cond]):


        present = sorted(f.name for f in d.glob("*.csv")) if d.exists() else []
        print(f"No accuracy/judge CSVs under {d} — nothing for this page to render.")
        print(f"  found: {', '.join(present) if present else '(no CSVs at all)'}")
        print(f"  needs: accuracy.csv / accuracy_delta.csv (from score_accuracy.py), and optionally"
              f" recall.csv, rating_distribution.csv")
        print(f"  ->  python scripts/score_accuracy.py --run {args.run} --split test")
        if (d / "test").exists() or (d / "val").exists():
            print(f"  NOTE: {d}/test and {d}/val exist. `--split` writes THERE, not here; this page"
                  f" reads the un-split copy, so run score_accuracy.py once with no --split too.")
        return 1

    colour = not args.no_colour and sys.stdout.isatty()
    prompts = prompt_drops(side) or prompt_drops(delta)

    print(f"\n{'='*78}\n  RUN: {args.run}   ({d})\n{'='*78}")
    print(f"  suppression prompts scored: {', '.join(prompts) if prompts else 'NONE — baseline only'}")

    print(f"\n{'='*78}\n  CROSS-MODEL SUMMARY — the whole result, one row per model x thinking x prompt type\n{'='*78}")
    print(_fmt(summary) if summary else "  (need accuracy_delta + answer_lengths — run score_accuracy + score_lengths)\n")
    if summary:
        import textwrap
        print(textwrap.fill(LEGEND, 96, initial_indent="  ", subsequent_indent="  ") + "\n")

    print(f"\n{'-'*78}\n  TABLE A — accuracy (general / MMLU arm)\n{'-'*78}")
    print(_fmt(delta))

    print(f"\n{'-'*78}\n  RQ4 — side-effect heatmap: suppressed concept x tested subject\n{'-'*78}")
    if not prompts:
        print("  (no suppression prompts scored — this is a baseline-only run)\n")
    for p in prompts:
        for model, ts, rows in _by_model_state(side):
            print(terminal_heatmap(rows, "concept_id", "mmlu_subject", f"{p}_drop",
                                   nk=f"{p}_n", title=f"  [{p}]  {model} · think = {ts}", colour=colour))

    print(f"\n{'-'*78}\n  SUBJECT × SUBJECT — accuracy: suppress concept (row) × test concept (col)\n{'-'*78}")
    if not matrix:
        print("  (subject_matrix.csv not found — re-run score_accuracy.py to generate it)\n")
    for p in prompt_drops(matrix):
        for model, ts, rows in _by_model_state(matrix):
            print(terminal_matrix(rows, f"{p}_acc", title=f"  [{p}]  {model} · think = {ts}"))

    print(f"\n{'-'*78}\n  TABLE A' — recall (isolated / RWKU arm)\n{'-'*78}")
    print(_fmt(recall))

    print(f"\n{'-'*78}\n  TABLE C — leakage + modes: PRIMARY judge (0-2 concept + framing)\n{'-'*78}")
    print(_fmt(rating))
    if rating:
        print("  leak_rate = P(concept_score > 0). strong_leak = P(concept_score == 2).\n"
              "  reasoned|framed|meta_leak are conditioned on the LEAKED rows, and are PROVISIONAL\n"
              "  until framing_coverage reaches 1.0 (an unframed leak counts as `leaked`).\n")
    print(f"\n{'-'*78}\n  cross-check: binary-primitive judge\n{'-'*78}")
    print(_fmt(modes))
    print(_fmt(cond))

    if args.html:
        out = d / "report.html"
        h = [f"<!doctype html><meta charset=utf-8><title>{args.run} — analysis</title>",
             f"<style>{CSS}</style>", f"<h1>{args.run} — analysis</h1>",
             f"<p class=meta>Rendered from <code>{d}</code>. Suppression prompts scored: "
             f"<b>{', '.join(prompts) if prompts else 'none — baseline only'}</b>.</p>",
             "<h2>Cross-model summary — the whole result on one page</h2>",
             html_table(summary, title="One row per model × thinking × prompt type", note=LEGEND)
             if summary else "<p class=empty>Run score_accuracy + score_lengths for the summary.</p>",
             "<h2>Table A — accuracy (general / MMLU arm)</h2>",
             html_table(delta, title="Baseline vs suppressed, per subject",
                        note="Paired per item on rows parsed in BOTH arms."),
             "<h2>RQ4 — side-effect heatmap</h2>"]
        if not prompts:
            h.append("<p class=empty>No suppression prompts scored — baseline-only run.</p>")
        for p in prompts:
            for model, ts, rows in _by_model_state(side):
                h.append(html_heatmap(rows, "concept_id", "mmlu_subject", f"{p}_drop",
                                      nk=f"{p}_n", title=f"{p} · {model} · think = {ts}"))
        h.append("<h2>Subject × subject — accuracy (suppress row, test col)</h2>")
        h.append(html_table(matrix, title="Concept×concept accuracy matrix",
                            note="Diagonal = on-target (suppress X, test X); off-diagonal = "
                                 "collateral. Re-run score_accuracy if empty.") if matrix
                 else "<p class=empty>subject_matrix.csv not scored — re-run score_accuracy.</p>")
        h += ["<h2>Table A' — recall (isolated / RWKU arm)</h2>",
              html_table(recall, title="RWKU token recall"),
              "<h2>Table C — leakage + modes (PRIMARY judge: 0–2 concept + framing)</h2>",
              html_table(rating, title="Leak rate + framing split",
                         note="leak_rate = P(concept_score &gt; 0); strong_leak = P(score == 2). "
                              "The reasoned/framed/meta split is conditioned on the leaked rows and "
                              "is PROVISIONAL until framing_coverage reaches 1.0 — an unframed leak "
                              "counts as `leaked`, so `meta_framed` grows as stage 2 runs."),
              "<h2>Cross-check — binary-primitive judge</h2>",
              html_table(modes, title="Mode distribution"),
              html_table(cond, title="P(leak | meta) — RQ3"),
              "<h2>Raw accuracy</h2>", html_table(acc, title="Per-cell accuracy + parse rate")]
        out.write_text("\n".join(h))
        print(f"\n  HTML -> {out}\n  (scp it back, or: python -m http.server --directory {d})")

    if args.save:
        from src.analysis.figures import (
            markdown_table, matrix_png, save_heatmap, table_png)

        dest = Path(args.save)
        written = []
        for w in (table_png(summary, dest / "cross_model_summary.png", title="Exp 1 — cross-model summary", note=LEGEND),
                  markdown_table(summary, dest / "cross_model_summary.md", title="Cross-model summary")):
            if w:
                written.append(w)
        for p in prompts:
            for model, ts, rows in _by_model_state(side):
                written += save_heatmap(rows, "concept_id", "mmlu_subject", f"{p}_drop",
                                        dest / f"heatmap_{p}_{model}_think-{ts}", nk=f"{p}_n",
                                        title=f"{args.run} · {p} · {model} · think = {ts}")
        if matrix:
            for p in prompt_drops(matrix):
                for model, ts, rows in _by_model_state(matrix):
                    written += matrix_png(rows, f"{p}_acc", dest / f"matrix_{p}_{model}_think-{ts}",
                                          title=f"{args.run} · {p} · {model} · think = {ts} · accuracy")
        else:
            print(f"  --save: NO matrix figure — {d / 'subject_matrix.csv'} is missing.\n"
                  f"          Re-run `python scripts/score_accuracy.py --run {args.run}` first "
                  "(it emits subject_matrix.csv), then re-run this.")
        for name, rows in (("accuracy_delta", delta), ("accuracy", acc), ("side_effect", side),
                           ("subject_matrix", matrix),
                           ("recall", recall), ("rating_distribution", rating),
                           ("mode_distribution", modes), ("conditional_stats", cond)):
            for w in (table_png(rows, dest / f"{name}.png", title=name),
                      markdown_table(rows, dest / f"{name}.md", title=name)):
                if w:
                    written.append(w)
        if not written:
            print(f"\n  --save: nothing to write yet (no scored tables under {d})")
        else:
            print(f"\n  saved {len(written)} file(s) -> {dest}/")
            for w in written:
                print(f"    {w.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
