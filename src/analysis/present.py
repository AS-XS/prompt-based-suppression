from __future__ import annotations

from typing import Iterable, Mapping, Sequence


PROBE_COLUMNS: tuple[str, ...] = ("forget", "neighbor", "retain", "utility")
PROBE_LABELS = {
    "forget": "Forget (on-target)",
    "neighbor": "Neighbor (near)",
    "retain": "Retain (other subj.)",
    "utility": "Utility (general)",
}
BASELINE_PROMPT = "none-baseline"

PROMPT_ORDER = (BASELINE_PROMPT, "world-frame", "ignorance-frame", "no-meta")
PROMPT_LABELS = {BASELINE_PROMPT: "Baseline"}


def prompts_in(rows: Iterable[Mapping]) -> list[str]:


    found = {c[: -len("_drop")] for r in rows for c in r if str(c).endswith("_drop")}
    known = [p for p in PROMPT_ORDER if p != BASELINE_PROMPT and p in found]
    return [BASELINE_PROMPT] + known + sorted(found - set(known))


_PLAIN_STATES = ("off", "na")


def _fmt_cell(acc: float | None, drop: float | None) -> str:


    if acc is None:
        return ""
    out = f"{acc * 100:.0f}%"
    if drop is not None:
        out += f" ({drop * 100:+.1f})"
    return out


def model_table(delta_rows: Iterable[Mapping], model: str,
                columns: Sequence[str] = PROBE_COLUMNS) -> tuple[list[str], list[list[str]]]:


    rows = [r for r in delta_rows if r.get("model") == model]
    present = [c for c in columns if any(r.get("probe_type") == c for r in rows)]
    header = ["Prompt Format"] + [PROBE_LABELS.get(c, c) for c in present]

    def cell(think: str, prompt: str, probe: str) -> str:
        match = [r for r in rows if r.get("think_state") == think and r.get("probe_type") == probe]
        if not match:
            return ""
        r = match[0]
        if prompt == BASELINE_PROMPT:
            b = r.get("baseline_acc")
            return _fmt_cell(float(b), None) if b not in ("", None) else ""
        acc, drop = r.get(f"{prompt}_acc"), r.get(f"{prompt}_drop")
        if acc in ("", None):
            return ""
        return _fmt_cell(float(acc), None if drop in ("", None) else float(drop))

    states = {r.get("think_state") for r in rows}
    blocks = [(s, "") for s in _PLAIN_STATES if s in states] + \
             ([("on", " (w/ think)")] if "on" in states else [])

    out: list[list[str]] = []
    for i, (think, suffix) in enumerate(blocks):
        if i:
            out.append([""] * len(header))          
        for p in prompts_in(rows):
            label = PROMPT_LABELS.get(p, p) + suffix
            line = [cell(think, p, c) for c in present]
            if any(line):
                out.append([label] + line)
    return header, out


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:

    width = len(header)

    def line(cells):
        padded = list(cells) + [""] * (width - len(cells))
        return "| " + " | ".join(padded) + " |"

    return "\n".join([line(header), "|" + "|".join([" :---- "] * width) + "|",
                      *(line(r) for r in rows)])


def render(delta_rows: Iterable[Mapping], *, title: str = "", note: str = "",
           models: Sequence[str] | None = None) -> str:

    rows = list(delta_rows)
    models = models or sorted({r.get("model") for r in rows if r.get("model")})
    parts = [f"# {title}" if title else "", note, ""]
    for m in models:
        header, body = model_table(rows, m)
        if not body:
            continue
        parts += [f"## {m}", "", markdown_table(header, body), ""]
    return "\n".join(p for p in parts if p is not None).strip() + "\n"


def concept_table(delta_rows: Iterable[Mapping], model: str, concept: str,
                  tested: Sequence[str] | None = None) -> tuple[list[str], list[list[str]]]:


    rows = [r for r in delta_rows
            if r.get("model") == model and r.get("concept_id") == concept]
    cols = list(tested) if tested else sorted({r.get("tested_concept") for r in rows
                                               if r.get("tested_concept")})
    header = ["Prompt Format"] + [f"{c} (on-target)" if c == concept else c for c in cols]

    def cell(think: str, prompt: str, tested_concept: str) -> str:
        match = [r for r in rows if r.get("think_state") == think
                 and r.get("tested_concept") == tested_concept]
        if not match:
            return ""
        r = match[0]
        if prompt == BASELINE_PROMPT:
            b = r.get("baseline_acc")
            return _fmt_cell(float(b), None) if b not in ("", None) else ""
        acc, drop = r.get(f"{prompt}_acc"), r.get(f"{prompt}_drop")
        if acc in ("", None):
            return ""
        return _fmt_cell(float(acc), None if drop in ("", None) else float(drop))

    states = {r.get("think_state") for r in rows}
    blocks = [(s, "") for s in _PLAIN_STATES if s in states] + \
             ([("on", " (w/ think)")] if "on" in states else [])

    out: list[list[str]] = []
    for i, (think, suffix) in enumerate(blocks):
        if i:
            out.append([""] * len(header))
        for p in prompts_in(rows):
            line = [cell(think, p, c) for c in cols]
            if any(line):
                out.append([PROMPT_LABELS.get(p, p) + suffix] + line)
    return header, out


def render_by_concept(delta_rows: Iterable[Mapping], model: str, *, title: str = "",
                      note: str = "", concepts: Sequence[str] | None = None) -> str:

    rows = [r for r in delta_rows if r.get("model") == model]
    concepts = concepts or sorted({r.get("concept_id") for r in rows if r.get("concept_id")})
    tested = sorted({r.get("tested_concept") for r in rows if r.get("tested_concept")})
    parts = [f"# {title or model}", note, ""]
    for c in concepts:
        header, body = concept_table(rows, model, c, tested=tested)
        if not body:
            continue
        parts += [f"## suppressing: {c}", "", markdown_table(header, body), ""]
    return "\n".join(p for p in parts if p is not None).strip() + "\n"
