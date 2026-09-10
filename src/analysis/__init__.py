from __future__ import annotations

import importlib


_EXPORTS = {
    "accuracy": ("accuracy_table", "delta_table", "extract_letter", "score_generations"),
    "dependency": ("compute_rows", "cross_subject_mention_fraction", "pairwise_matrix",
                   "subtree_overlap", "tree_distance"),
    "modes": ("MODES", "conditional_stats", "derive_mode", "mode_distribution", "mode_of"),
    "surfaces": ("answer_of", "split_think", "trace_of"),
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
