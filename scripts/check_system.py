#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_SYSTEM_TEXT = "SUPPRESSION_INSTRUCTION_SENTINEL"
_USER_TEXT = "PROBE_QUESTION_SENTINEL"
_GLUE = "\n\n"  


def _render(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def classify(tok) -> tuple[str, str]:

    try:
        with_system = _render(tok, [{"role": "system", "content": _SYSTEM_TEXT},
                                    {"role": "user", "content": _USER_TEXT}])
    except Exception as e:
        return "REJECTED", f"{type(e).__name__}: {str(e)[:90]}"

    if _SYSTEM_TEXT not in with_system:
        return "DROPPED", "the system content is absent from the rendered prompt"


    glued = _render(tok, [{"role": "user", "content": _SYSTEM_TEXT + _GLUE + _USER_TEXT}])
    if with_system == glued:
        return "MERGED", "renders identically to exp2's single user turn"


    between = with_system.split(_SYSTEM_TEXT, 1)[1].split(_USER_TEXT, 1)[0]
    return "SEPARATE", f"delimiter between turns: {between.strip()[:60]!r}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*",
                    help='Names.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--show", action="store_true",
                    help='Show.')
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not importable — run this in the venv (source venv/bin/activate)")
        return 1

    root = Path(args.models_dir)
    dirs = [root / n for n in args.names] if args.names else sorted(
        d for d in root.iterdir() if d.is_dir())

    print(f"{'model':30} {'verdict':10} detail")
    verdicts: dict[str, str] = {}
    for d in dirs:
        if not d.is_dir():
            print(f"{d.name:30} -- no such dir under {root}")
            continue
        try:
            tok = AutoTokenizer.from_pretrained(str(d), trust_remote_code=True)
        except Exception as e:
            print(f"{d.name:30} {'ERROR':10} {type(e).__name__}: {str(e)[:70]}")
            continue
        if not getattr(tok, "chat_template", None):
            print(f"{d.name:30} {'NONE':10} no chat template at all")
            continue
        verdict, detail = classify(tok)
        verdicts[d.name] = verdict
        flag = "" if verdict == "SEPARATE" else "  <-- CHECK"
        print(f"{d.name:30} {verdict:10} {detail}{flag}")

        if args.show:
            rendered = _render(tok, [{"role": "system", "content": _SYSTEM_TEXT},
                                     {"role": "user", "content": _USER_TEXT}])
            print(f"\n===== {d.name} — rendered [system, user] =====")
            print(rendered[:1200])
            print("=" * 60 + "\n")

    bad = {k: v for k, v in verdicts.items() if v != "SEPARATE"}
    print()
    if bad:
        print("EXCLUDE FROM EXP-3 (or the rows will measure nothing):")
        for k, v in sorted(bad.items()):
            print(f"  {k:30} {v}")
        print("\nMERGED is not a crash — it produces a full, plausible results file in which exp3 is")
        print("a duplicate of exp2. Drop those models from the exp3 model list rather than letting")
        print("them dilute the position contrast.")
    elif verdicts:


        names = ", ".join(sorted(verdicts))
        scope = f"the {len(verdicts)} model(s) checked ({names})"
        print(f"SEPARATE for {scope} — the exp3 contrast is well-defined there.")
        if args.names:
            print("\n⚠️  This was a SUBSET — only the names you passed. Every model you intend to")
            print("    sweep must appear in one of these runs before you spend GPU on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
