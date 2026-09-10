from __future__ import annotations

from pathlib import Path

from src.analysis.report import MIN_N, pivot, pivot_n, spread


def _require_mpl():
    try:
        import matplotlib
    except ModuleNotFoundError:
        raise SystemExit(
            "--save needs matplotlib, which is an optional extra (it is deliberately not a hard\n"
            "dependency: report.py must stay stdlib-only so it runs on the login node).\n"
            "  pip install 'matplotlib>=3.8'      # or: pip install -e '.[analysis]'\n"
            "Without it, `--html` gives you the same heatmap in a browser and needs nothing.")
    matplotlib.use("Agg")   
    import matplotlib.pyplot as plt
    return plt


def save_heatmap(rows, rk, ck, vk, out_stem: Path, *, nk=None, title="",
                 min_n: int = MIN_N) -> list[Path]:


    plt = _require_mpl()
    rowkeys, colkeys, cells = pivot(rows, rk, ck, vk)
    if not cells:
        return []
    ns = pivot_n(rows, rk, ck, nk) if nk else None
    limit = max(abs(v) for v in cells.values())

    grid = [[cells.get((r, c)) for c in colkeys] for r in rowkeys]
    fig, ax = plt.subplots(figsize=(1 + 0.62 * len(colkeys), 1.4 + 0.42 * len(rowkeys)))
    im = ax.imshow([[0 if v is None else v * 100 for v in row] for row in grid],
                   cmap="RdBu_r", vmin=-limit * 100, vmax=limit * 100, aspect="auto")

    ax.set_xticks(range(len(colkeys)))
    ax.set_xticklabels([c.replace("high_school_", "hs_").replace("college_", "col_")
                        for c in colkeys], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rowkeys)))
    ax.set_yticklabels(rowkeys, fontsize=8)
    ax.set_xlabel("tested subject", fontsize=9)
    ax.set_ylabel("suppressed concept", fontsize=9)

    for i, r in enumerate(rowkeys):
        for j, c in enumerate(colkeys):
            v = cells.get((r, c))
            if v is None:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0f0f0",
                                           edgecolor="none"))
                continue
            small = ns is not None and ns.get((r, c), 0) < min_n
            if small:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch="///",
                                           edgecolor="#00000055", linewidth=0))
            ax.text(j, i, f"{v*100:+.0f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > limit * .55 else "black")

    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.02)
    cb.set_label("accuracy drop (percentage points)\n+ = suppression cost accuracy", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    s = spread(rowkeys, colkeys, cells, ns, min_n)
    cap = (f"spread across rows (which concept is suppressed): {s['row_spread']*100:.1f} pp   ·   "
           f"across columns (which subject is tested): {s['col_spread']*100:.1f} pp")
    if s["excluded"]:
        cap += f"\nhatched = n < {min_n}, shown but excluded from the spread"
    ax.set_title(title, fontsize=10, pad=10)
    fig.text(.01, .005, cap, fontsize=7, color="#444")
    fig.tight_layout(rect=(0, .04, 1, 1))

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        p = out_stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight", **kw)
        written.append(p)
    plt.close(fig)
    return written


def matrix_png(rows: list[dict], vk: str, out_stem: Path, *, title="") -> list[Path]:


    from src.analysis.report import matrix_summary

    plt = _require_mpl()
    s = matrix_summary(rows, vk)
    rk, ck, cells = s["rk"], s["ck"], s["cells"]
    if not cells:
        return []
    base = s.get("baseline") or {}


    BL = "baseline (none)"
    ylabels = [BL] + rk

    def cell_at(rl, c):
        return base.get(c) if rl == BL else cells.get((rl, c))

    grid = [[cell_at(rl, c) for c in ck] for rl in ylabels]
    fig, ax = plt.subplots(figsize=(1.6 + 0.95 * len(ck), 1.4 + 0.7 * len(ylabels)))
    im = ax.imshow([[(0 if v is None else v * 100) for v in row] for row in grid],
                   cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(ck)))
    ax.set_xticklabels(ck, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.get_yticklabels()[0].set_fontweight("bold")     
    ax.set_xlabel("tested concept", fontsize=9)
    ax.set_ylabel("suppressed concept (top row = none)", fontsize=9)
    ax.axhline(0.5, color="#111111", linewidth=1.6)    
    for i, rl in enumerate(ylabels):
        for j, c in enumerate(ck):
            v = cell_at(rl, c)
            if v is None:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0f0f0",
                                           edgecolor="none"))
                continue
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center", fontsize=8,
                    color="black")
            if rl != BL and rl == c:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor="#111111", linewidth=2.2))
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
    cb.set_label("accuracy under suppression (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)


    cap = ("top row = baseline (no suppression); read each column down from it.   "
           f"on-target (boxed diagonal) drop {s['diag']*100:+.1f} pp  vs  collateral (off-diagonal) "
           f"{s['off']*100:+.1f} pp" if s["diag"] is not None and s["off"] is not None else "")
    ax.set_title(title, fontsize=10, pad=10)
    if cap:
        fig.text(.01, .005, cap, fontsize=7.5, color="#444")
    fig.tight_layout(rect=(0, .04, 1, 1))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, kw in ((".pdf", {}), (".png", {"dpi": 200})):
        p = out_stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight", **kw)
        written.append(p)
    plt.close(fig)
    return written


_OUTCOME_STYLE: dict[str, tuple[str, str]] = {
    "held":           ("#2e7d32", "held — suppression worked (forget)"),
    "intact":         ("#a5d6a7", "intact — untouched (off-target)"),
    "leaked_correct": ("#c62828", "leaked_correct — knowledge still used (forget)"),
    "collateral":     ("#ef6c00", "collateral — off-target damage"),
    "unparsed":       ("#9e9e9e", "unparsed — nothing scoreable"),
}
_OUTCOME_ORDER = ("held", "intact", "leaked_correct", "collateral", "unparsed")
_PROBE_ORDER = ("forget", "neighbor", "retain", "utility")   


def outcome_png(rows: list[dict], out: Path, *, title: str = "", note: str = "") -> Path | None:


    if not rows:
        return None
    plt = _require_mpl()

    probes = [p for p in _PROBE_ORDER if any(r.get("probe_type") == p for r in rows)]
    probes += sorted({str(r.get("probe_type")) for r in rows} - set(probes) - {"None"})
    ykeys = sorted({(str(r.get("model") or ""), str(r.get("think_state") or ""),
                     str(r.get("prompt_id") or "")) for r in rows})
    if not probes or not ykeys:
        return None
    by = {(str(r.get("model") or ""), str(r.get("think_state") or ""),
           str(r.get("prompt_id") or ""), str(r.get("probe_type"))): r for r in rows}

    fig, axes = plt.subplots(1, len(probes), sharey=True,
                             figsize=(1.6 + 3.1 * len(probes), 1.9 + 0.34 * len(ykeys)))
    axes = list(axes) if len(probes) > 1 else [axes]
    ypos = list(range(len(ykeys)))

    for ax, probe in zip(axes, probes):
        for y, key in zip(ypos, ykeys):
            r = by.get((*key, probe))
            if not r:
                continue
            eligible = int(r.get("n_eligible") or 0)
            if not eligible:
                ax.text(1, y, " no eligible rows", va="center", fontsize=6.5, color="#999")
                continue
            left = 0.0
            for oc in _OUTCOME_ORDER:
                n = int(r.get(oc) or 0)
                if not n:
                    continue
                frac = 100 * n / eligible
                ax.barh(y, frac, left=left, height=0.72, color=_OUTCOME_STYLE[oc][0],
                        edgecolor="white", linewidth=0.4)
                if frac >= 11:
                    ax.text(left + frac / 2, y, f"{frac:.0f}", ha="center", va="center",
                            fontsize=6.5, color="white", fontweight="bold")
                left += frac
            ax.text(101.5, y, f"n={eligible}", va="center", fontsize=6, color="#666")
        ax.set_xlim(0, 100)
        ax.set_title(probe + (" (on-target)" if probe == "forget" else ""), fontsize=9,
                     fontweight="bold" if probe == "forget" else "normal")
        ax.set_xlabel("% of eligible rows", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#eeeeee", linewidth=0.6)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([f"{m} · {p}" + (f"  [think {t}]" if t and t != "na" else "")
                             for m, t, p in ykeys], fontsize=7)
    axes[0].invert_yaxis()

    for i in range(1, len(ykeys)):
        if ykeys[i][0] != ykeys[i - 1][0]:
            for ax in axes:
                ax.axhline(i - 0.5, color="#cccccc", linewidth=0.8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=_OUTCOME_STYLE[o][0]) for o in _OUTCOME_ORDER]
    fig.legend(handles, [_OUTCOME_STYLE[o][1] for o in _OUTCOME_ORDER], loc="lower center",
               ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.012))
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.0)
    cap = ("Each row is compared with ITS OWN baseline on the SAME item, so a bar counts only what "
           "suppression changed. `already_wrong` (the baseline never got it right) is excluded from "
           "both the bars and n — including it would dilute the rate.")
    fig.text(0.005, -0.055, (note + "  " if note else "") + cap, fontsize=6.8, color="#444",
             wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


_ON, _OFF = "on-target (forget)", "off-target (retain · neighbor · utility)"
_FAMILY = {_ON: ("held", "leaked_correct"), _OFF: ("intact", "collateral")}
_CLEAN_STATES = {"off", "na"}


def summary_cells(rows: list[dict]) -> tuple[list[str], list[str], dict, str]:


    clean = [r for r in rows if str(r.get("think_state") or "") in _CLEAN_STATES]
    arm_note = "thinking OFF only (think-on accuracy is truncation-confounded)"
    if not clean:
        clean, arm_note = list(rows), "ALL reasoning states — this run has no thinking-off arm"

    prompts = sorted({str(r.get("prompt_id")) for r in clean if r.get("prompt_id")})
    models = sorted({str(r.get("model")) for r in clean if r.get("model")})
    cells: dict[tuple[str, str], tuple[dict[str, float], int, int]] = {}
    for prompt in prompts:
        for family, (good, bad) in _FAMILY.items():
            per_model, eligible = [], 0
            for m in models:
                rs = [r for r in clean if str(r.get("model")) == m
                      and str(r.get("prompt_id")) == prompt
                      and ((str(r.get("probe_type")) == "forget") == (family == _ON))]
                n = sum(int(r.get("n_eligible") or 0) for r in rs)
                if not n:
                    continue
                eligible += n
                per_model.append({k: 100 * sum(int(r.get(k) or 0) for r in rs) / n
                                  for k in (good, bad, "unparsed")})
            cells[(prompt, family)] = (
                ({k: sum(d[k] for d in per_model) / len(per_model) for k in per_model[0]}
                 if per_model else {}), eligible, len(per_model))
    return prompts, models, cells, arm_note


def outcome_summary_png(rows: list[dict], out: Path, *, title: str = "",
                        note: str = "") -> Path | None:


    if not rows:
        return None
    plt = _require_mpl()

    prompts, models, cells, arm_note = summary_cells(rows)
    if not prompts:
        return None

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11.5, 1.7 + 0.55 * len(prompts)))
    ypos = list(range(len(prompts)))
    for ax, family in zip(axes, (_ON, _OFF)):
        good, bad = _FAMILY[family]
        for y, prompt in zip(ypos, prompts):
            pcts, eligible, _ = cells[(prompt, family)]
            if not pcts:
                continue
            left = 0.0
            for oc in (good, bad, "unparsed"):
                v = pcts.get(oc, 0.0)
                if v <= 0:
                    continue
                ax.barh(y, v, left=left, height=0.62, color=_OUTCOME_STYLE[oc][0],
                        edgecolor="white", linewidth=0.6)
                if v >= 9:
                    ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                            color="white", fontweight="bold")
                left += v
            ax.text(101.5, y, f"n={eligible:,}", va="center", fontsize=6.5, color="#777")


        ax.set_title(f"{family}\nwant: {_OUTCOME_STYLE[good][1].split(' — ')[0]}", fontsize=9,
                     fontweight="bold" if family == _ON else "normal")
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of rows the baseline got right", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#eeeeee", linewidth=0.6)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([p.replace("none-baseline", "baseline") for p in prompts], fontsize=9)
    axes[0].invert_yaxis()

    seen = [o for fam in _FAMILY.values() for o in fam] + ["unparsed"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=_OUTCOME_STYLE[o][0]) for o in seen]
    fig.legend(handles, [_OUTCOME_STYLE[o][1] for o in seen], loc="lower center", ncol=3,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.0)
    cap = (f"{arm_note}.  {len(models)} model(s), MACRO-averaged — each contributes equally, so a "
           "model with more finished waves cannot outvote one with fewer.  Rows the baseline got "
           "wrong are excluded: there was nothing there for suppression to remove.  Per-model and "
           "per-probe-type detail is in outcomes.png.")
    fig.text(0.005, -0.075, (note + "  " if note else "") + cap, fontsize=7, color="#444",
             wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def markdown_table(rows: list[dict], out: Path, *, title="") -> Path | None:

    if not rows:
        return None
    cols = list(rows[0])
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


def table_png(rows: list[dict], out: Path, *, title="", note="") -> Path | None:


    if not rows:
        return None
    import textwrap
    plt = _require_mpl()
    cols = list(rows[0])
    body = [[str(r.get(c, "")) for c in cols] for r in rows]

    wrapped = textwrap.fill(note, width=int(11 * len(cols)) or 110) if note else ""
    note_lines = wrapped.count("\n") + 1 if wrapped else 0

    fig_w = min(2.2 + 1.15 * len(cols), 26)
    fig_h = min(0.8 + 0.30 * (len(body) + 1) + 0.22 * note_lines, 60)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11, pad=10, loc="left", fontweight="bold")
    if wrapped:
        fig.text(0.01, 0.01, wrapped, fontsize=7.5, color="#333333", va="bottom", wrap=True)

    tbl = ax.table(cellText=body, colLabels=cols, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    for (r, _c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f7f7f7")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out
