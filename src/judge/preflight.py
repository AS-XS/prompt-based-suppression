from __future__ import annotations

from pathlib import Path
from typing import Callable


def load_tokenizer(model_dir: str | Path):

    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_dir))
    except Exception as e:
        print(f"  preflight: no tokenizer at {model_dir} ({type(e).__name__}) — the thinking "
              f"default and the window check are SKIPPED", flush=True)
        return None


def think_default(render: Callable[..., str]) -> str:


    try:
        bare = render()
        on, off = render(enable_thinking=True), render(enable_thinking=False)
    except Exception:
        return "unknown"
    if on == off:
        return "no toggle"
    return "on" if bare == on else "off" if bare == off else "unknown"


def window_violations(lengths: list[int], limit: int | None) -> list[int]:

    if not limit:
        return []
    return [i for i, n in enumerate(lengths) if n >= limit]


def run_preflight(tok, conversations: list[list[dict]], limit: int | None, labels=None) -> str:


    if tok is None:
        return "unchecked"

    def render(msgs, **kw):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    probe = conversations[0] if conversations else [{"role": "user", "content": "x"}]
    default = think_default(lambda **kw: render(probe, **kw))
    print(f"  preflight: chat template's own thinking default = {default} "
          f"(nothing is passed; the judge runs at it)", flush=True)
    lengths = [len(tok(render(c), add_special_tokens=False)["input_ids"]) for c in conversations]
    if lengths:
        print(f"  preflight: {len(lengths):,} prompt(s), longest {max(lengths):,} tokens, "
              f"window {limit if limit else 'unknown'}", flush=True)
    bad = window_violations(lengths, limit)
    if bad:
        names = labels or [str(i) for i in range(len(conversations))]
        shown = "\n".join(f"    {names[i]}  {lengths[i]:,} tokens" for i in bad[:20])
        raise SystemExit(f"{len(bad)} prompt(s) leave no room to answer in the judge's "
                         f"{limit:,}-token window:\n{shown}\n  Drop those rows from the input; the "
                         f"window is the model's own and is not shrunk or truncated here.")
    return default
