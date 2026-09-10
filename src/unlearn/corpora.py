from __future__ import annotations

from pathlib import Path


_FORGET = {
    "wmdp-cyber": ("cais/wmdp-corpora", "cyber-forget-corpus"),
    "wmdp-bio": ("cais/wmdp-corpora", "bio-forget-corpus"),      
}


_RETAIN = ("Salesforce/wikitext", "wikitext-103-raw-v1")

GATED = {"wmdp-bio"}


def load_forget(concept_id: str, *, limit: int | None = None,
                local_path: str | Path | None = None) -> list[str]:


    if local_path:
        return _read_local(Path(local_path), limit)
    if concept_id not in _FORGET:
        raise SystemExit(f"no forget corpus registered for {concept_id!r}; "
                         f"known: {sorted(_FORGET)}")
    name, config = _FORGET[concept_id]
    try:
        from datasets import load_dataset
        ds = load_dataset(name, config, split="train")
    except Exception as e:
        extra = ""
        if concept_id in GATED:
            extra = ("\n  This corpus is GATED. Request access at https://huggingface.co/datasets/"
                     "cais/wmdp-corpora, then pass --forget-file <the downloaded file>.\n"
                     "  Do NOT substitute other text: a reproduction trained on a different corpus "
                     "produces a number that looks like RMU's and is not.")
        raise SystemExit(
            f"could not load {name}/{config}: {e}{extra}\n"
            f"  If this says 'offline' or 'connection', the corpus is simply not cached yet and the "
            f"compute node cannot fetch it . Prefetch the corpora on a network-enabled machine:\n"
            f"      python scripts/unlearn.py --fetch-corpora --method rmu --concept {concept_id}")
    texts = [t for t in ds["text"] if t and t.strip()]
    return texts[:limit] if limit else texts


def load_retain(*, limit: int | None = None) -> list[str]:

    try:
        from datasets import load_dataset
        ds = load_dataset(*_RETAIN, split="train")
    except Exception as e:
        raise SystemExit(
            f"could not load {_RETAIN[0]}/{_RETAIN[1]}: {e}\n"
            f"  Prefetch unavailable corpora on a network-enabled machine:\n"
            f"      python scripts/unlearn.py --fetch-corpora --method rmu --concept wmdp-cyber")
    texts = [t for t in ds["text"] if t and len(t.strip()) > 50]
    return texts[:limit] if limit else texts


def _read_local(path: Path, limit: int | None) -> list[str]:
    import json

    if not path.exists():
        raise SystemExit(f"no such forget file: {path}")
    out: list[str] = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                out.append(line)
                continue
            t = row.get("text") if isinstance(row, dict) else None
            if t:
                out.append(t)
    else:
        out = [p for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    if not out:
        raise SystemExit(f"{path} yielded no text")
    return out[:limit] if limit else out
