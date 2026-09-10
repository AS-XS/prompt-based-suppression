#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent / "models.txt"


def roster() -> list[str]:
    out = []
    for line in MANIFEST.read_text().splitlines():
        repo = line.split("#", 1)[0].strip()
        if repo:
            out.append(repo)
    return out


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())


def _indexed_shards(d: Path) -> set[str]:
    for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = d / idx
        if p.exists():
            return set(json.loads(p.read_text()).get("weight_map", {}).values())
    return set()


def check(d: Path) -> tuple[str, str]:
    if not d.is_dir() or not any(d.iterdir()):
        return "MISSING", "no dir / empty"
    if any(d.rglob("*.incomplete")):
        return "INCOMPLETE", "has .incomplete files (interrupted download)"
    if not (d / "config.json").exists():
        return "INCOMPLETE", "no config.json"

    shards = _indexed_shards(d)
    if shards:
        missing = [s for s in shards if not (d / s).exists() or (d / s).stat().st_size == 0]
        if missing:
            return "INCOMPLETE", f"missing {len(missing)}/{len(shards)} weight shards"
    else:
        weights = [w for w in [*d.glob("*.safetensors"), *d.glob("*.bin")] if w.stat().st_size]
        if not weights:
            return "INCOMPLETE", "no weight files"

    has_tok = any((d / f).exists() for f in
                  ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "vocab.json"))
    return "OK", _human(_dir_size(d)) + ("" if has_tok else "  (WARN: no tokenizer)")


def check_online(repo: str, d: Path, note: str) -> tuple[str, str]:
    try:
        from huggingface_hub import HfApi

        remote = [f for f in HfApi().list_repo_files(repo)
                  if not f.endswith((".md", ".gitattributes")) and not f.startswith("original/")]
        missing = [f for f in remote if not (d / f).exists()]
        if missing:
            return "INCOMPLETE", f"{note} — missing {len(missing)} repo file(s), e.g. {missing[0]}"
    except Exception as e:  # noqa: BLE001 — online check is best-effort
        return "OK", f"{note}  (online check skipped: {e})"
    return "OK", note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--online", action="store_true")
    args = ap.parse_args()

    root = Path(args.models_dir)
    print(f"models dir: {root}\n")

    repos = roster()
    bad = 0
    for repo in repos:
        d = root / repo.split("/")[-1]
        status, note = check(d)
        if args.online and status == "OK":
            status, note = check_online(repo, d, note)
        bad += status != "OK"
        print(f"  [{status:10}] {repo.split('/')[-1]:34} {note}")

    print()
    if bad:
        print(f"{bad}/{len(repos)} model(s) NEED ATTENTION — "
              f"fix with: bash scripts/download_all.sh")
    else:
        print(f"ALL GOOD — {len(repos)} models complete.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
