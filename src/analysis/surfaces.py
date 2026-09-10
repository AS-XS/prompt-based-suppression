from __future__ import annotations

import re


TAG_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),        
    ("<|channel>", "<channel|>"),   
)


_CONTROL_NOISE = ("<turn|>", "<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>",
                  "<|im_end|>", "<|endoftext|>", "<|end|>")


def _clean(text: str) -> str:
    for tok in _CONTROL_NOISE:
        text = text.replace(tok, "")
    return text.strip()


def split_think(raw: str | None, *, expect_trace: bool = False) -> tuple[str | None, str]:


    if not raw:
        return None, ""
    for _open, _close in TAG_PAIRS:
        if _open not in raw:


            if _close in raw:
                trace, _, answer = raw.partition(_close)
                return _clean(trace), _clean(answer)
            continue
        traces: list[str] = []
        answer, rest = "", raw
        while _open in rest:
            head, _, after = rest.partition(_open)
            answer += head
            if _close not in after:
                traces.append(after)
                return _clean("\n".join(traces)), ""
            trace, _, rest = after.partition(_close)
            traces.append(trace)
        return _clean("\n".join(traces)), _clean(answer + rest)


    return None, ("" if expect_trace else _clean(raw))


def answer_of(raw: str | None, *, expect_trace: bool = False) -> str:


    return split_think(raw, expect_trace=expect_trace)[1]


def trace_of(raw: str | None) -> str | None:

    return split_think(raw)[0]


_REASON = re.compile(r"<reason>(.*?)(?:</reason>|$)", re.IGNORECASE | re.DOTALL)
_WHY = re.compile(r"\b(?:why|reasoning|reason|explanation|rationale)\b[\s*_`~\-–]*:", re.IGNORECASE)
_ANSWER_LINE = re.compile(r"^\s*\**\s*answer\b[\s:*_`~\-–]*\(?\s*[A-D]\b[\)\.:,\s]*", re.IGNORECASE)


def explanation_of(raw: str | None) -> str:


    ans = answer_of(raw)
    if not ans:
        return ""


    m = _REASON.search(ans)
    if m:
        return m.group(1).strip()
    m = _WHY.search(ans)
    if m:
        return ans[m.end():].strip()
    lines = ans.splitlines()
    if lines and _ANSWER_LINE.match(lines[0]):
        return "\n".join(lines[1:]).strip()
    return ans.strip()
