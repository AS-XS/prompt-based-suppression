from src.analysis.surface_gap import leak_pattern, leaks, novel_pattern
from src.analysis.surfaces import explanation_of, trace_of


def lexical_flags(r: dict) -> dict:


    cid, cls, pid = r["concept_id"], r["concept_class"], r["prompt_id"]
    pat, npat = leak_pattern(cid, cls), novel_pattern(pid, cid, cls)
    tr, ex = trace_of(r["raw_output"]) or "", explanation_of(r["raw_output"]) or ""
    has_ex = bool(ex.strip())
    return {"lex_trace": leaks(tr, pat),
            "lex_trace_novel": leaks(tr, npat) if npat is not None else None,
            "lex_explanation": leaks(ex, pat) if has_ex else None,
            "lex_explanation_novel": leaks(ex, npat) if (has_ex and npat is not None) else None,
            "trace_chars": len(tr), "has_explanation": has_ex}
