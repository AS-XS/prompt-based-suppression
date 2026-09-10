from __future__ import annotations

from src.gen.thinkstate import think_states_for


_THINKING_MIN = 90.0
_QUIET_MIN = 9.0
_BASE_ITEMS = 4040


_MEASURED: dict[str, float] = {
    "Olmo-3-7B-Think": 291.0,     
}


_SIZE = (("1.7B", 0.4), ("1.5B", 0.4), ("E2B", 0.5), ("3B", 0.6), ("4B", 0.7),
         ("7B", 1.0), ("8B", 1.0), ("9B", 1.2), ("10B", 1.3), ("12B", 1.6),
         ("13B", 1.7), ("14B", 2.0), ("20B", 2.4), ("24B", 3.0), ("27B", 3.4), ("32B", 4.0))


def _size_factor(model: str) -> float:
    for tag, mult in _SIZE:
        if tag.lower() in model.lower():
            return mult
    return 1.0


def _token_scale(max_tokens: int) -> float:


    return 1.0 + 0.4 * max(0.0, (max_tokens / 4096) - 1.0)


def minutes_per_wave(model: str, *, n_items: int = _BASE_ITEMS, max_tokens: int = 4096) -> float:

    scale = (n_items / _BASE_ITEMS) * _token_scale(max_tokens)
    if model in _MEASURED:
        return _MEASURED[model] * scale * len(think_states_for(model))
    per_state = _size_factor(model)
    total = sum((_THINKING_MIN if s == "on" else _QUIET_MIN) * per_state
                for s in think_states_for(model))
    return total * scale


def waves_per_job(model: str, *, budget_hours: float = 9.0, n_items: int = _BASE_ITEMS,
                  max_tokens: int = 4096) -> int:


    per = minutes_per_wave(model, n_items=n_items, max_tokens=max_tokens)
    return max(1, int(budget_hours * 60 // per)) if per > 0 else 1
