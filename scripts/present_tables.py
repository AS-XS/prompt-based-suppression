#!/usr/bin/env python


from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import read_dicts as _read_dicts  # noqa: E402
from src.analysis.accuracy import delta_table, score_generations  # noqa: E402
from src.analysis.grouping import eval_subject, tested_concept  # noqa: E402
from src.analysis.present import (concept_table, model_table, render,  # noqa: E402
                                  render_by_concept)
from src.data.mmlu import SUBJECT_TO_CONCEPT  # noqa: E402
from src.paths import generations_glob, items_path, presentation_dir  # noqa: E402


_CSS = """
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:2rem auto;
max-width:1100px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.6rem}h2{margin-top:2.5rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
h3{margin-top:1.6rem;font-size:1rem;color:#333}
.sub{color:#555;font-size:.85rem;background:#f8f9fa;padding:.6rem .8rem;border-radius:4px}
table{border-collapse:collapse;margin:.5rem 0 1.5rem;font-variant-numeric:tabular-nums}
th,td{padding:.35rem .7rem;border-bottom:1px solid #eee;text-align:right}
th{background:#f5f5f5;font-size:.8rem}
th:first-child,td:first-child{text-align:left;white-space:nowrap}
tr.sep td{border:0;height:.5rem}
img.fig{max-width:100%;height:auto;display:block;margin:1rem 0 1.5rem;border:1px solid #eee;
border-radius:4px;background:#fff}
@media(prefers-color-scheme:dark){body{background:#161616;color:#e8e8e8}
 h2{border-color:#333}h3{color:#bbb}th{background:#222}th,td{border-color:#2a2a2a}
 .sub{background:#1e1e1e;color:#bbb}img.fig{border-color:#333}}
"""


def _html_table(header, body) -> str:

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    out = ["<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in header) + "</tr>"]
    for r in body:
        cells = list(r) + [""] * (len(header) - len(r))
        cls = " class=sep" if not any(str(c).strip() for c in cells) else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>")
    return "\n".join(out) + "</table>"


def _png(header, body, out: Path, *, title: str, note: str = "") -> Path | None:


    from src.analysis.figures import table_png

    rows = [dict(zip(header, r)) for r in body]
    out.parent.mkdir(parents=True, exist_ok=True)
    return table_png(rows, out, title=title, note=note)


_NOTE_TMPL = (
    "Cells are accuracy, with the item-paired drop from that model's own baseline in parentheses "
    "(`+` = suppression cost accuracy). Columns are probe types **pooled over {n} suppressed "
    "{noun}**{why}. A blank cell means not scored, never zero."
)
_WHY_MMLU = (
    ", so `Forget` and `Retain` are drawn from the same balanced mix of subjects — subjects differ a "
    "lot in how fragile they are under any unusual prompt (mathematics ~-28 pp vs biology ~-4 pp "
    "when suppressing *anything*), and pooling is what cancels that out"
)


def _note(scored) -> str:

    concepts = {s.get("concept_id") for s in scored
                if s.get("probe_type") == "forget" and s.get("concept_id")}
    n = len(concepts)

    mmlu = n >= 4 and not any(str(c).startswith("wmdp") for c in concepts)
    return _NOTE_TMPL.format(n=n or "the", noun="concept" if n == 1 else "concepts",
                             why=_WHY_MMLU if mmlu else "")


_POOLED_WARNING = (
    "\n\n> ⚠️ **The `Forget` column here pools {n} areas with different headroom** ({spread}). "
    "Pooling cancels per-subject noise only when the mix is balanced, which it is not here — read "
    "the per-concept tables in `by_concept/`, not this cell, for anything WMDP."
)


def _pooled_warning(scored) -> str:

    base: dict[str, list[bool]] = {}
    for s in scored:
        if s.get("probe_type") == "forget" and s.get("prompt_id") == "none-baseline" and s.get("parsed"):
            base.setdefault(s.get("mmlu_subject") or "?", []).append(bool(s.get("correct")))
    if len(base) < 2:
        return ""
    rates = {k: sum(v) / len(v) for k, v in base.items() if v}
    if max(rates.values()) - min(rates.values()) < 0.10:
        return ""
    spread = ", ".join(f"{k} {r * 100:.0f}%" for k, r in sorted(rates.items(), key=lambda kv: -kv[1]))
    return _POOLED_WARNING.format(n=len(rates), spread=spread)

_ISOLATED_NOTE = (
    "The **isolated** arm: RWKU entities, free text, no multiple choice — entity items only. Cells "
    "are **token-recall** (the share of the gold answer's tokens that appear), with the item-paired "
    "drop from baseline in parentheses (`+` = suppression cost recall). `Forget` = questions about "
    "the suppressed entity (on-target); `Neighbor` = questions about closely related entities "
    "(collateral). Read a low BASELINE as a warning, not a result: if the model never knew the "
    "entity, suppression has nothing to suppress and a small drop means ignorance rather than "
    "success. **Not token-F1/EM:** RWKU items are asked with no answer-format instruction, so the "
    "model answers in prose against a 2-3 token gold — F1's precision term then measures answer "
    "LENGTH rather than knowledge, and rises when a prompt merely makes the model terser. F1/EM "
    "become meaningful once these items are prompted for a short answer. A blank cell means not "
    "scored, never zero."
)

_ERRORS_NOTE = (
    "Not accuracy — <b>what changed, item by item</b>. Every suppressed answer is compared with the "
    "same model's own baseline answer to the same question, so a bar counts only what suppression "
    "actually altered; questions the model got wrong anyway are excluded, because there was nothing "
    "there to suppress. <b>Left panel is the goal</b> (the suppressed concept's own questions — we "
    "WANT dark green there); the right panel is everything else, where we want it left alone. "
    "Suppression that behaved looks like a dark-green left panel beside a pale-green right one."
)

_BY_CONCEPT_NOTE = (
    "One table per suppressed concept: columns are the individual **tested** concepts, so the "
    "on-target column sits beside every retain subject rather than one chosen comparison. Read the "
    "spread across a row before calling a retain drop 'spillover' — subjects differ a lot in how "
    "fragile they are under any unusual prompt (mathematics ~-28 pp vs biology ~-4 pp when "
    "suppressing *anything*), so a large drop in one retain column may be that subject, not the "
    "suppression. The pooled table in `../tables.md` is the summary of these."
)


def _item_index(path: Path) -> dict[str, dict]:
    idx = {}
    for it in _read_dicts(path):
        meta = it.get("meta") or {}
        idx[it["item_id"]] = {"answer_key": it.get("answer_key"), "format": meta.get("format"),
                              "mmlu_subject": eval_subject(it),
                              "probe_type": it.get("probe_type"), "concept_id": it.get("concept_id"),

                              "reference": it.get("reference"),


                              "concept_subtype": it.get("concept_subtype")}
    return idx


def _isolated_rows(gens, idx, metric: str = "recall"):


    from src.analysis.recall import recall_delta, score_recall

    entities = {k: v for k, v in idx.items() if v.get("concept_subtype") == "entity"}
    scored = score_recall(gens, entities)
    if not scored:
        return []
    rows, prompts = recall_delta(scored, by=("model", "think_state", "probe_type"), metric=metric)
    out = []
    for r in rows:
        n = {"model": r["model"], "think_state": r["think_state"], "probe_type": r["probe_type"],
             "baseline_acc": r.get("baseline"), "n_baseline": r.get("n_baseline")}
        for p in prompts:
            n[f"{p}_acc"], n[f"{p}_drop"], n[f"{p}_n"] = r.get(p), r.get(f"{p}_drop"), r.get(f"{p}_n")
        out.append(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp1")
    ap.add_argument("--items", default=None, help='Items.')
    ap.add_argument("--generations", nargs="+", default=None, help='Generations.')
    ap.add_argument("--models", nargs="+", default=None, help='Models.')
    ap.add_argument("--baseline-model", default=None,
                    help='Baseline model.')
    ap.add_argument("--isolated-metric", default="recall", choices=["recall", "hit", "f1", "em"],
                    help='Isolated metric.')
    ap.add_argument("--out", default=None, help='Out.')
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-by-concept", action="store_true",
                    help='No by concept.')
    ap.add_argument("--no-html", action="store_true",
                    help='No html.')
    ap.add_argument("--no-png", action="store_true",
                    help='No png.')
    args = ap.parse_args()

    items_file = Path(args.items) if args.items else items_path(args.run)
    if not items_file.exists():
        raise SystemExit(f"no item set at {items_file} — build it first "
                         f"(python -m src.data.build --run {args.run} ...)")
    gen_globs = args.generations or [generations_glob(args.run)]
    paths = [f for pat in gen_globs for f in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no generation files matched: {gen_globs}")

    idx = _item_index(items_file)
    gens = [g for p in paths for g in _read_dicts(Path(p))]
    scored = score_generations(gens, idx)
    if not scored:
        raise SystemExit("no MC generations scored — are there MC items (meta.format=='mc') "
                         "and do the item_ids match?")


    if args.baseline_model:
        if not any(s_["model"] == args.baseline_model for s_ in scored):
            raise SystemExit(
                f"--baseline-model {args.baseline_model!r} has no rows in this run — it is the "
                f"reference half of a paired comparison and must be swept into THIS namespace with "
                f"the same config.\n  present: {sorted({s_['model'] for s_ in scored})}")
        from src.analysis.checkpoints import refuse_cross_family
        refuse_cross_family((s_["model"] for s_ in scored), args.baseline_model)
        rows, edited = delta_table(scored, by=("think_state", "probe_type"),
                                   baseline_prompt=args.baseline_model, contrast="model")


        rows = [{**r, "model": m} for r in rows for m in edited
                if r.get(f"{m}_acc") not in (None, "")]
    else:
        rows, _ = delta_table(scored, by=("model", "think_state", "probe_type"))

    out = Path(args.out) if args.out else presentation_dir(args.run) / "tables.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    pooled_note = _note(scored) + _pooled_warning(scored)
    text = render(rows, title=args.title or f"{args.run} — results by model", note=pooled_note,
                  models=args.models)
    out.write_text(text, encoding="utf-8")
    models = sorted(args.models or {r["model"] for r in rows})
    print(f"wrote {len(models)} pooled model table(s) -> {out}")


    iso = _isolated_rows(gens, idx, metric=args.isolated_metric)
    iso_label = {"recall": "token-recall", "hit": "hit rate", "f1": "token-F1",
                 "em": "exact match"}[args.isolated_metric]
    iso_out = out.parent / "isolated.md"
    if iso:
        iso_out.write_text(render(iso, title=f"{args.run} — isolated arm (RWKU, {iso_label})",
                                  note=_ISOLATED_NOTE, models=args.models), encoding="utf-8")
        print(f"wrote isolated (RWKU) table(s) -> {iso_out}")
    else:
        print("  (no isolated table — no RWKU items with a gold `reference` matched these generations)")


    errors_png = None
    if not args.no_png:
        try:
            from src.analysis.errors import outcome_table, pair_outcomes
            from src.analysis.figures import outcome_summary_png

            paired = pair_outcomes(scored)
            if paired:
                otab = outcome_table(paired, by=("model", "think_state", "prompt_id", "probe_type"))
                errors_png = outcome_summary_png(
                    otab, out.parent / "png" / "errors_summary.png",
                    title=f"{args.run} — did suppression bite on target, or everywhere?")
                if errors_png:
                    print(f"wrote error summary figure -> {errors_png}")
            else:
                print("  (no error summary — no suppressed rows could be paired with a baseline)")
        except ImportError:
            pass

    by_concept, tested = [], []
    pngs = 0
    if not args.no_png:
        try:
            for m in models:
                header, body = model_table(rows, m)
                if body and _png(header, body, out.parent / "png" / f"{m}.png",
                                 title=m, note=pooled_note):
                    pngs += 1
                ih, ib = model_table(iso, m)
                if ib and _png(ih, ib, out.parent / "png" / f"{m}__isolated.png",
                               title=f"{m} — isolated (RWKU, {iso_label})", note=_ISOLATED_NOTE):
                    pngs += 1
        except ImportError:
            print("  (no PNGs — matplotlib missing; pip install -e '.[analysis]')")

    if not args.no_by_concept:


        for s in scored:
            s["tested_concept"] = tested_concept(s, SUBJECT_TO_CONCEPT)
        by_concept, _ = delta_table(
            [s for s in scored if s.get("tested_concept")],
            by=("model", "think_state", "concept_id", "tested_concept"))
        dest = out.parent / "by_concept"
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        tested = sorted({r.get("tested_concept") for r in by_concept if r.get("tested_concept")})
        for m in models:
            text = render_by_concept(by_concept, m, title=f"{m} — by suppressed concept",
                                     note=_BY_CONCEPT_NOTE)
            if text.count("## ") == 0:
                continue
            (dest / f"{m}.md").write_text(text, encoding="utf-8")
            n += text.count("## ")
            if args.no_png:
                continue
            concepts = sorted({r["concept_id"] for r in by_concept
                               if r.get("model") == m and r.get("concept_id")})
            try:
                for c in concepts:
                    h, b = concept_table(by_concept, m, c, tested=tested)
                    if b and _png(h, b, dest / "png" / f"{m}__{c}.png",
                                  title=f"{m} — suppressing: {c}", note=_BY_CONCEPT_NOTE):
                        pngs += 1
            except ImportError:
                pass
        print(f"wrote {n} per-concept table(s) -> {dest}/<model>.md")
    if pngs:
        print(f"wrote {pngs} PNG(s) -> {out.parent}/png/ and {out.parent}/by_concept/png/")

    if not args.no_html:


        from src.analysis.present import markdown_table, model_table as _mt
        parts = [f"<h1>{args.run} — results</h1>"]
        if errors_png:


            import base64

            b64 = base64.b64encode(Path(errors_png).read_bytes()).decode("ascii")
            parts += ["<h2>The summary — what suppression actually did</h2>",
                      f"<p class=sub>{_ERRORS_NOTE}</p>",
                      f"<img class=fig src='data:image/png;base64,{b64}' alt='error outcome summary'>"]
        parts += ["<h2>General arm (MMLU) — accuracy</h2>",
                  f"<p class=sub>{_note(scored)}</p>"]
        for m in models:
            h, b = _mt(rows, m)
            if b:
                parts += [f"<h3>{m}</h3>", _html_table(h, b)]
        parts += ["<h2>Isolated arm (RWKU) — token-F1</h2>", f"<p class=sub>{_ISOLATED_NOTE}</p>"]
        for m in models:
            h, b = _mt(iso, m)
            if b:
                parts += [f"<h3>{m}</h3>", _html_table(h, b)]
        if not args.no_by_concept:
            parts += ["<h2>Per suppressed concept</h2>", f"<p class=sub>{_BY_CONCEPT_NOTE}</p>"]
            for m in models:
                cs = sorted({r["concept_id"] for r in by_concept
                             if r.get("model") == m and r.get("concept_id")})
                for c in cs:
                    h, b = concept_table(by_concept, m, c, tested=tested)
                    if b:
                        parts += [f"<h3>{m} — suppressing: {c}</h3>", _html_table(h, b)]
        html = out.parent / "results.html"
        html.write_text(f"<!doctype html><meta charset=utf-8><title>{args.run} — results</title>"
                        f"<style>{_CSS}</style>" + "\n".join(parts), encoding="utf-8")
        print(f"wrote one-page HTML -> {html}")
    print("  (analysis/ untouched — these are separate deliverables)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
