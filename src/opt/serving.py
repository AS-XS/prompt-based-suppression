from __future__ import annotations

from src.analysis.surfaces import trace_of
from src.gen.thinkstate import chat_kwargs, needs_raw_special, think_states_for


def extra_body(model: str, think_state: str) -> dict:


    body: dict = {}
    kw = chat_kwargs(model, think_state).get("chat_template_kwargs")
    if kw:
        body["chat_template_kwargs"] = kw
    if needs_raw_special(model):
        body["skip_special_tokens"] = False
    return body


def check_think_state(model: str, think_state: str, text: str) -> str | None:


    if think_state not in think_states_for(model):
        return (f"{model} does not sweep think_state={think_state!r} "
                f"(it has {think_states_for(model)}) — the baseline wave you are pairing against "
                f"cannot exist")
    if think_state == "na":
        return None
    has_trace = bool(trace_of(text))
    if think_state == "off" and has_trace:
        return ("asked for think-OFF and the model emitted a reasoning block. The toggle is not "
                "reaching the server, so every generation is in the wrong state and the metric is "
                "pairing think-on answers against think-off baselines.")
    if think_state == "on" and not has_trace:
        return ("asked for think-ON and no reasoning block came back. Either the toggle is not "
                "reaching the server, or the delimiters are control tokens the server stripped "
                "(see `needs_raw_special`) — in which case the trace is silently inside the answer.")
    return None
