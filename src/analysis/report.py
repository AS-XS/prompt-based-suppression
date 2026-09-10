from __future__ import annotations

import csv
from pathlib import Path


_RED = (178, 24, 43)
_BLUE = (33, 102, 172)
_WHITE = (247, 247, 247)


def load(path: str | Path) -> list[dict]:


    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def prompt_drops(rows: list[dict]) -> list[str]:

    if not rows:
        return []
    return sorted(c[: -len("_drop")] for c in rows[0] if c.endswith("_drop"))


def pivot(rows: list[dict], rk: str, ck: str, vk: str) -> tuple[list[str], list[str], dict]:

    cells: dict[tuple[str, str], float] = {}
    for r in rows:
        v = r.get(vk, "")
        if v in ("", None):
            continue
        cells[(r[rk], r[ck])] = float(v)
    return (sorted({r for r, _ in cells}), sorted({c for _, c in cells}), cells)


def pivot_n(rows: list[dict], rk: str, ck: str, nk: str) -> dict:


    ns: dict[tuple[str, str], int] = {}
    for r in rows:
        v = r.get(nk, "")
        if v in ("", None):
            continue
        ns[(r[rk], r[ck])] = int(float(v))
    return ns


def _lerp(a: tuple, b: tuple, t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def heat_rgb(v: float, limit: float) -> tuple[int, int, int]:

    if limit <= 0:
        return _WHITE
    t = max(-1.0, min(1.0, v / limit))
    return _lerp(_WHITE, _RED if t > 0 else _BLUE, abs(t))


def _ink(rgb: tuple[int, int, int]) -> tuple[int, int, int]:

    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (0, 0, 0) if lum > 140 else (255, 255, 255)


def _cell(v: float | None, limit: float, w: int, colour: bool) -> str:
    if v is None:
        return " " * w
    txt = f"{v * 100:+.1f}".rjust(w)
    if not colour:
        return txt
    bg, fg = heat_rgb(v, limit), _ink(heat_rgb(v, limit))
    return f"\x1b[48;2;{bg[0]};{bg[1]};{bg[2]}m\x1b[38;2;{fg[0]};{fg[1]};{fg[2]}m{txt}\x1b[0m"


MIN_N = 40
"""Cells below this many items are excluded from the spread. Not arbitrary: the utility columns
carry ~15 items, so their accuracy moves in steps of 1/15 = 6.7 pp and every value in them is a
multiple of 6.7. Pooling those with the ~100-item STEM columns inflated the measured column spread
from 12.8 pp to 20.0 pp — i.e. the headline RQ4 number became mostly quantisation noise."""


def spread(rowkeys, colkeys, cells, ns: dict | None = None, min_n: int = MIN_N) -> dict:


    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    if ns is None:
        keep, excluded = cells, None
    else:
        keep = {k: v for k, v in cells.items() if ns.get(k, 0) >= min_n}
        excluded = sorted({c for (r, c) in cells if ns.get((r, c), 0) < min_n})
        rowkeys = [r for r in rowkeys if any((r, c) in keep for c in colkeys)]
        colkeys = [c for c in colkeys if any((r, c) in keep for r in rowkeys)]

    rmeans = {r: mean([keep.get((r, c)) for c in colkeys]) for r in rowkeys}
    cmeans = {c: mean([keep.get((r, c)) for r in rowkeys]) for c in colkeys}
    rv = [v for v in rmeans.values() if v is not None]
    cv = [v for v in cmeans.values() if v is not None]
    return {
        "row_means": rmeans, "col_means": cmeans, "excluded": excluded, "min_n": min_n,
        "row_spread": (max(rv) - min(rv)) if rv else 0.0,
        "col_spread": (max(cv) - min(cv)) if cv else 0.0,
    }


_PREFIXES = (("high_school_", "hs"), ("college_", "col"), ("professional_", "prof"))


def _split_label(name: str) -> tuple[str, str]:
    for full, short in _PREFIXES:
        if name.startswith(full):
            return short, name[len(full):]
    return "", name


def terminal_heatmap(rows, rk, ck, vk, *, nk=None, title="", colour=True) -> str:

    rowkeys, colkeys, cells = pivot(rows, rk, ck, vk)
    ns = pivot_n(rows, rk, ck, nk) if nk else None
    if not cells:
        return f"{title}\n  (no cells — the suppression waves have not run)\n"
    limit = max(abs(v) for v in cells.values())
    w = 6
    lab = max((len(r) for r in rowkeys), default=10) + 2

    pre, stem = zip(*(_split_label(c) for c in colkeys))
    out = [f"\n{title}",
           f"  cells = accuracy drop in pp  (+ = suppression cost accuracy)  ·  scale ±{limit*100:.1f} pp",
           " " * lab + "".join(p.rjust(w) for p in pre),
           " " * lab + "".join(s[: w - 1].rjust(w) for s in stem)]
    for r in rowkeys:
        line = r.ljust(lab) + "".join(_cell(cells.get((r, c)), limit, w, colour) for c in colkeys)
        out.append(line)

    s = spread(rowkeys, colkeys, cells, ns)
    out += [
        "",
        f"  spread across ROWS ({rk}, i.e. which concept is suppressed): {s['row_spread']*100:5.1f} pp",
        f"  spread across COLS ({ck}, i.e. which subject is tested)    : {s['col_spread']*100:5.1f} pp"
        + ("   <-- DOMINANT: reads as non-specific degradation, not a blast radius"
           if s["col_spread"] > s["row_spread"] else ""),
    ]
    if s["excluded"]:
        out.append(f"  (spread excludes {len(s['excluded'])} column(s) under n={s['min_n']} — "
                   f"{', '.join(s['excluded'])} — where accuracy moves in steps of ~6.7 pp and the "
                   f"marginal mean is mostly quantisation noise)")
    elif ns is None:
        out.append("  (spread over ALL cells — no item counts supplied, so small noisy cells are "
                   "pooled with large ones; pass nk= to filter)")
    return "\n".join(out) + "\n"


def html_heatmap(rows, rk, ck, vk, *, nk=None, title="") -> str:
    rowkeys, colkeys, cells = pivot(rows, rk, ck, vk)
    ns = pivot_n(rows, rk, ck, nk) if nk else None
    if not cells:
        return f"<h3>{title}</h3><p class=empty>No cells — the suppression waves have not run.</p>"
    limit = max(abs(v) for v in cells.values())
    h = [f"<h3>{title}</h3>",
         f"<p class=sub>Accuracy drop in percentage points. + = suppression cost accuracy. "
         f"Scale ±{limit*100:.1f} pp, centred on 0.</p>",
         "<table class=heat><tr><th></th>"]
    h += [f"<th>{c.replace('high_school_','hs_').replace('college_','col_')}</th>" for c in colkeys]
    h.append("</tr>")
    for r in rowkeys:
        h.append(f"<tr><th class=rowlab>{r}</th>")
        for c in colkeys:
            v = cells.get((r, c))
            if v is None:
                h.append("<td class=na></td>")
                continue
            bg, fg = heat_rgb(v, limit), _ink(heat_rgb(v, limit))
            h.append(f'<td style="background:rgb{bg};color:rgb{fg}" title="{r} suppressed / '
                     f'{c} tested">{v*100:+.1f}</td>')
        h.append("</tr>")
    h.append("</table>")
    s = spread(rowkeys, colkeys, cells, ns)
    dom = s["col_spread"] > s["row_spread"]
    h.append(
        f"<p class={'warn' if dom else 'sub'}>Spread across rows ({rk}, which concept is suppressed): "
        f"<b>{s['row_spread']*100:.1f} pp</b> &nbsp;·&nbsp; across columns ({ck}, which subject is "
        f"tested): <b>{s['col_spread']*100:.1f} pp</b>"
        + (" — the column effect dominates, which reads as non-specific degradation rather than a "
           "concept-specific blast radius (RQ4)." if dom else "") + "</p>")
    if s["excluded"]:
        h.append(f"<p class=sub>Spread excludes {len(s['excluded'])} column(s) with fewer than "
                 f"{s['min_n']} items (<i>{', '.join(s['excluded'])}</i>) — there accuracy moves in "
                 f"steps of ~6.7 pp, so the marginal mean is mostly quantisation noise. The cells "
                 f"are still shown above.</p>")
    return "\n".join(h)


def matrix_summary(rows: list[dict], vk: str) -> dict:


    rk, ck, cells = pivot(rows, "concept_id", "tested_concept", vk)
    _, _, dcells = pivot(rows, "concept_id", "tested_concept", vk.rsplit("_", 1)[0] + "_drop")
    diag = [dcells[(c, c)] for c in rk if (c, c) in dcells]
    off = [v for (r, c), v in dcells.items() if r != c]


    bsum: dict[str, list[float]] = {}
    for r in rows:
        y, b, n = r.get("tested_concept"), r.get("baseline_acc"), r.get("n_baseline")
        if y in ck and b not in ("", None):
            w = float(n) if n not in ("", None) else 0.0
            bsum.setdefault(y, [0.0, 0.0])
            bsum[y][0] += float(b) * w
            bsum[y][1] += w
    baseline = {y: (num / den if den else None) for y, (num, den) in bsum.items()}

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    return {"rk": rk, "ck": ck, "cells": cells, "baseline": baseline,
            "diag": mean(diag), "off": mean(off)}


def terminal_matrix(rows: list[dict], vk: str, *, title="", pct=True) -> str:


    s = matrix_summary(rows, vk)
    rk, ck, cells = s["rk"], s["ck"], s["cells"]
    if not cells:
        return f"{title}\n  (no cells — subject_matrix.csv not scored; re-run score_accuracy)\n"
    lab = max(max(len(r) for r in rk), len("baseline (none)")) + 2
    short = {c: c[:6] for c in ck}
    out = [f"\n{title}", "  row = SUPPRESSED concept · col = TESTED concept · * = on-target diagonal",
           " " * lab + "".join(f"{short[c]:>8}" for c in ck)]

    base = s.get("baseline") or {}
    bline = "baseline (none)".ljust(lab)
    for c in ck:
        b = base.get(c)
        bline += (" " * 8 if b is None else (f"{b*100:6.0f} " if pct else f"{b:6.2f} "))
    out += [bline, " " * lab + "".join("  ----- "[:8] for _ in ck)]
    for r in rk:
        line = r.ljust(lab)
        for c in ck:
            v = cells.get((r, c))
            if v is None:
                line += " " * 8
            else:
                mark = "*" if r == c else " "
                line += (f"{v*100:6.0f}{mark}" if pct else f"{v:6.2f}{mark}")
        out.append(line)
    if s["diag"] is not None and s["off"] is not None:
        gap = s["diag"] - s["off"]                       
        verdict = ("   (on-target DROP larger → concept-specific)" if gap > 0.02
                   else "   (≈ equal → suppression is NON-SPECIFIC, not a blast radius)")
        out += ["",
                "  grid = accuracy %; the verdict below is read off the DROP (accuracy conflates the",
                "  forget vs retain baselines, so raw accuracy alone fakes a concept-specific gap):",
                f"  on-target (diagonal) mean DROP : {s['diag']*100:+5.1f} pp",
                f"  collateral (off-diag) mean DROP: {s['off']*100:+5.1f} pp",
                f"  on-target − collateral         : {gap*100:+5.1f} pp" + verdict]
    return "\n".join(out) + "\n"


def html_table(rows: list[dict], *, title="", note="") -> str:
    if not rows:
        return f"<h3>{title}</h3><p class=empty>Not scored yet.</p>"
    cols = list(rows[0])
    h = [f"<h3>{title}</h3>"]
    if note:
        h.append(f"<p class=sub>{note}</p>")
    h.append("<div class=scroll><table class=data><tr>")
    h += [f"<th>{c}</th>" for c in cols]
    h.append("</tr>")
    for r in rows:
        h.append("<tr>" + "".join(f"<td>{r.get(c,'')}</td>" for c in cols) + "</tr>")
    h.append("</table></div>")
    return "\n".join(h)


CSS = """
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:2rem auto;
max-width:1200px;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.5rem;margin-bottom:.2rem}h2{margin-top:2.5rem;border-bottom:2px solid #eee;
padding-bottom:.3rem}h3{margin-top:1.8rem;font-size:1.05rem}
.sub{color:#666;font-size:.85rem;margin:.2rem 0 .8rem}
.empty{color:#999;font-style:italic}
.warn{color:#a11;font-size:.85rem;background:#fff5f5;padding:.5rem .7rem;border-left:3px solid #a11}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
.heat td{text-align:right;padding:.35rem .5rem;font-size:.82rem;min-width:3.4rem}
.heat th{font-size:.72rem;padding:.3rem;text-align:right;font-weight:600;color:#444}
.heat th.rowlab{text-align:left;padding-right:.7rem;white-space:nowrap}
.heat td.na{background:#fafafa}
.data{font-size:.8rem;width:100%}
.data th{background:#f5f5f5;padding:.4rem .5rem;text-align:left;border-bottom:1px solid #ddd;
position:sticky;top:0}
.data td{padding:.3rem .5rem;border-bottom:1px solid #f0f0f0}
.data tr:hover td{background:#fafcff}
.scroll{overflow-x:auto;max-height:32rem;overflow-y:auto;border:1px solid #eee}
.meta{background:#f8f9fa;padding:.8rem 1rem;border-radius:4px;font-size:.82rem;color:#555}
.meta code{background:#fff;padding:.1rem .3rem;border-radius:3px}
@media(prefers-color-scheme:dark){
 body{background:#161616;color:#e8e8e8}h2{border-color:#333}
 .data th{background:#222;border-color:#333}.data td{border-color:#262626}
 .data tr:hover td{background:#1e2430}.scroll{border-color:#2a2a2a}
 .heat td.na{background:#1c1c1c}.meta{background:#1e1e1e;color:#bbb}.meta code{background:#2a2a2a}
 .warn{background:#2a1a1a}}
"""
