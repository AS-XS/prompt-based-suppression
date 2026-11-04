from __future__ import annotations

from pathlib import Path


def layer_window(spec: str, depth: int) -> tuple[int, int]:


    if spec == "auto":
        spec = "rel:4-8"
    if spec.startswith("rel:"):
        a, b = map(float, spec[4:].split("-"))
        start, end = round(depth * a / 32), round(depth * b / 32)
    else:
        start, end = map(int, spec.split(","))
    if not 0 <= start < end <= depth:
        raise ValueError(f"invalid layer range {spec!r} for {depth} layers")
    return start, end


def run_elm(*_args, **_kwargs):

    raise SystemExit(
        "ELM is not reimplemented here (src/unlearn/elm.py explains why: its objective is a "
        "three-distribution guided loss, and a subtly wrong version produces a plausible number "
        "that is not ELM's).\n"
        "  Use scripts/fetch_elm.py then scripts/train_elm.py (pinned author code), or\n"
        "  run the authors' code externally (github.com/rohitgandikota/erasing-llm), then\n"
        "  measure the checkpoint through OUR pipeline so the comparison is on our items:\n"
        "      python scripts/unlearn.py --method elm --from-checkpoint <path> --name elm-llama-bio\n"
        "  That registers it and prints the sweep command; nothing else differs from the RMU path.")


def register_checkpoint(path: str | Path, name: str, out_root: str | Path = "models/unlearned") -> Path:


    src, dst = Path(path).resolve(), Path(out_root) / name
    if not src.exists():
        raise SystemExit(f"no such checkpoint: {src}")
    if not (src / "config.json").exists():
        raise SystemExit(f"{src} has no config.json — is it a HF checkpoint directory?")
    if dst.exists() or dst.is_symlink():
        raise SystemExit(f"{dst} already exists — pick another --name, or remove it first")
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            (dst / f.name).symlink_to(f)
    print(f"registered {src}  ->  {dst}  ({len(list(dst.iterdir()))} files linked)")
    return dst
