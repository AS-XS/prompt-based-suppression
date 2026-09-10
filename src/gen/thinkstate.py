from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from src.paths import UNLEARN_REPORT, UNLEARNED_ROOT, unlearn_report_path

_REGISTRY = Path(__file__).resolve().parents[2] / "configs" / "think_states.yaml"


@lru_cache(maxsize=1)
def registry() -> dict:
    import yaml

    return yaml.safe_load(_REGISTRY.read_text(encoding="utf-8")) or {}


def _chat_template(model_dir: Path) -> str | None:


    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8").strip()
    cfg = model_dir / "tokenizer_config.json"
    if cfg.is_file():
        try:
            t = json.loads(cfg.read_text(encoding="utf-8")).get("chat_template")
        except (OSError, json.JSONDecodeError):
            return None
        return t.strip() if isinstance(t, str) else None
    return None


@lru_cache(maxsize=None)
def _inherited(model_name: str, root: str) -> tuple[str, dict] | None:


    for d in dict.fromkeys((Path(root) / model_name, UNLEARNED_ROOT / model_name)):
        if unlearn_report_path(d).is_file():
            break
    else:
        return None
    report = unlearn_report_path(d)
    try:
        meta = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"{report} is unreadable ({e}) — it is the only record of which model "
                         f"{model_name!r} was edited from, so its think-states cannot be derived. "
                         f"Repair or delete the checkpoint.")
    base = meta.get("base_model")
    if not base or base not in registry():
        return None
    src = Path(meta.get("model") or "")
    ours, theirs = _chat_template(d), _chat_template(src) if src.is_dir() else None
    if theirs is not None and ours != theirs:
        raise SystemExit(
            f"{model_name!r} was edited from {base!r}, but their chat templates DIFFER — so its "
            f"reasoning capability cannot be inherited (the template is what opens a think block).\n"
            f"  checkpoint {d}\n  base       {src}\n"
            f"Audit the checkpoint and give it its own entry:\n"
            f"  MODELS_DIR={root} python scripts/check_think.py {model_name} --show")
    return base, registry()[base]


def _entry(model_name: str) -> dict:
    reg = registry()
    if model_name in reg:
        return reg[model_name]
    inherited = _inherited(model_name, os.environ.get("MODELS_DIR", "./models"))
    if inherited is not None:
        return inherited[1]
    raise KeyError(
        f"{model_name!r} is not in configs/think_states.yaml — its reasoning capability is "
        f"UNKNOWN, and guessing is exactly what this registry exists to prevent (a model with a "
        f"thinking toggle swept as 'na' produces mislabelled rows; if its toggle defaults ON, "
        f"silently-thinking ones).\n"
        f"  Audit it:  MODELS_DIR=./models python scripts/check_think.py {model_name} --show\n"
        f"  then add an entry: states / toggle / tag / default.\n"
        f"  (An unlearned checkpoint inherits its base model's entry instead — that needs a "
        f"{UNLEARN_REPORT} beside it naming a `base_model` that IS registered.)"
    )


def inherited_from(model_name: str) -> str | None:


    got = _inherited(model_name, os.environ.get("MODELS_DIR", "./models"))
    return got[0] if got and model_name not in registry() else None


def think_states_for(model_name: str) -> list[str]:

    return list(_entry(model_name)["states"])


def chat_kwargs(model_name: str, think_state: str) -> dict:


    entry = _entry(model_name)
    toggle = entry.get("toggle")
    if not toggle or think_state == "na":
        return {}
    if toggle == "enable_thinking":
        return {"chat_template_kwargs": {"enable_thinking": think_state == "on"}}

    return {}


def needs_raw_special(model_name: str) -> bool:


    return bool(_entry(model_name).get("control_tokens"))


def think_tag(model_name: str) -> str | None:


    return _entry(model_name).get("tag")
