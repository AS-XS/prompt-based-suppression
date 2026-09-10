from __future__ import annotations

import importlib

_EXPORTS = {
    "prompts": ("concept_messages", "framing_messages", "primitive_messages"),
    "run": ("extract_rating", "parse_primitives", "run_judge", "run_primitive_judge"),
    "schema": ("JudgeOutput", "JudgmentRow", "judge_json_schema"),
}
_NAME_TO_MODULE = {name: mod for mod, names in _EXPORTS.items() for name in names}
__all__ = sorted(_NAME_TO_MODULE)


def __getattr__(name: str):
    mod = _NAME_TO_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{mod}"), name)


def __dir__() -> list[str]:
    return __all__
