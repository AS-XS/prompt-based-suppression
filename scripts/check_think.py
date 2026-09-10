#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gen.thinkstate import think_states_for  # noqa: E402


_TOGGLES = ("enable_thinking", "add_generation_prompt and thinking", "reasoning_effort",
            "enable_reasoning", "thinking_mode")


_TAGS = ("<think>", "</think>", "<|think|>", "<|/think|>", "<|end_think|>", "<|think_end|>",
         "<|channel>thought", "<reasoning>", "</reasoning>", "<|thinking|>", "<start_working_out>")


def _template(model_dir: Path) -> str:

    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        return jinja.read_text(encoding="utf-8", errors="replace")
    cfg = model_dir / "tokenizer_config.json"
    if cfg.exists():
        try:
            t = json.loads(cfg.read_text(encoding="utf-8", errors="replace")).get("chat_template")
        except Exception:
            return ""
        if isinstance(t, list):
            return "\n".join(str(x.get("template", x)) for x in t)
        return t or ""
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help='Names.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--audit", metavar="OUT", default=None,
                    help='Audit.')
    ap.add_argument("--show", action="store_true",
                    help='Show.')
    args = ap.parse_args()

    root = Path(args.models_dir)
    dirs = [root / n for n in args.names] if args.names else sorted(
        d for d in root.iterdir() if d.is_dir())

    print(f"{'model':30} {'we sweep':10} {'toggle in template':22} {'think tags':12} verdict")
    for d in dirs:
        if not d.is_dir():
            print(f"{d.name:30} -- no such dir under {root}")
            continue
        tpl = _template(d)
        if not tpl:
            print(f"{d.name:30} {str(think_states_for(d.name)):10} (no chat template found)")
            continue
        toggles = [k for k in _TOGGLES if k in tpl]
        tags = [k for k in _TAGS if k in tpl]
        ours = think_states_for(d.name)

        if toggles:
            verdict = "HAS A TOGGLE -> sweep off/on?" if ours == ["na"] else "ok"
        elif tags:
            verdict = "REASONS, no toggle -> 'on', not 'na'" if ours == ["na"] else "ok"
        else:
            verdict = "ok (no thinking) " if ours == ["na"] else "?? we sweep thinking, template has none"
        flag = "" if verdict.startswith("ok") else "  <-- CHECK"
        print(f"{d.name:30} {str(ours):10} {str(toggles or '-'):22} {str(tags or '-'):12} {verdict}{flag}")

    if args.show:
        for d in dirs:
            if not d.is_dir():
                continue
            tpl = _template(d)
            hits = [(i, ln.strip()) for i, ln in enumerate(tpl.splitlines(), 1)
                    if any(k in ln for k in _TOGGLES + _TAGS + ("thinking", "think"))]
            print(f"\n===== {d.name} — every template line mentioning think/thinking =====")
            if not hits:
                print("   (none)")
            for i, ln in hits[:40]:
                print(f"  {i:4}| {ln[:150]}")
            if len(hits) > 40:
                print(f"   ... {len(hits)-40} more")

    if args.audit:
        lines = [
            "# Per-model reasoning capability -- READ OFF THE CHAT TEMPLATE, not vendor docs or memory.",
            "# The template is what vLLM applies and what decides whether a think block opens.",
            "#",
            "# GENERATED DRAFT (scripts/check_think.py --audit). Every `default:` below is UNKNOWN until",
            "# a human reads the template (`--show`) -- a regex can see that a toggle EXISTS, not what it",
            "# does when we pass nothing. A toggle defaulting ON makes every 'na' row a thinking row.",
            "#",
            "#   states  : think_states to sweep",
            "#   toggle  : chat_template_kwargs key that switches it (null = no runtime switch)",
            "#   tags    : the markers the template mentions (surfaces.py splits on <think>)",
            "#   default : what happens when we pass nothing -- FILL IN BY READING, then verify by",
            "#             generating 2 rows and checking whether a think block appears",
            "",
        ]
        for d in dirs:
            if not d.is_dir():
                continue
            tpl = _template(d)
            toggles = [k for k in _TOGGLES if k in tpl]
            tags = [k for k in _TAGS if k in tpl]
            if toggles:
                states = "[off, on]"
            elif tags:
                states = "[on]"
            else:
                states = "[na]"
            lines += [
                f"{d.name}:",
                f"  states: {states}",
                f"  toggle: {toggles[0] if toggles else 'null'}",
                f"  tags: {tags or '[]'}",
                f"  default: UNKNOWN        # <- read the template, then confirm empirically",
                f"  we_sweep_today: {think_states_for(d.name)}",
                "",
            ]
        Path(args.audit).write_text("\n".join(lines))
        print(f"\nDRAFT registry -> {args.audit}  ({sum(1 for d in dirs if d.is_dir())} models)")
        print("  every `default:` says UNKNOWN on purpose: a regex proves a toggle exists, not what")
        print("  it does by default. Read them with --show, fill them in, then send it to Claude.")

    print("\nA toggle/tag on a model we sweep as 'na' means its rows are MISLABELLED: think_state says")
    print("'na' while the model reasoned. surfaces.py still splits the <think> block, so the answer")
    print("stays correct -- but the RQ2 contrast would be comparing against a silently always-on model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
