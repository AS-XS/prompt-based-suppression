#!/usr/bin/env python3

import hashlib
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.elm_upstream import REVISION, SOURCE_SHA256  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", action="store_true", help='Corpora.')
    args = ap.parse_args()
    root = Path("data/external/elm")
    root.mkdir(parents=True, exist_ok=True)
    for remote, local in [("trainscripts/erase.py", "erase.py"),
                          ("data/wmdp-keywords.json", "wmdp-keywords.json")]:
        url = f"https://raw.githubusercontent.com/rohitgandikota/erasing-llm/{REVISION}/{remote}"
        content = urllib.request.urlopen(url, timeout=60).read()
        if local == "erase.py" and hashlib.sha256(content).hexdigest() != SOURCE_SHA256:
            raise SystemExit("ELM source hash mismatch")
        target = root / local
        if target.exists() and target.read_bytes() != content:
            raise SystemExit(f"refusing to overwrite different source: {target}")
        target.write_bytes(content)
        print(f"{target}: {hashlib.sha256(content).hexdigest()}")
    if args.corpora:
        from datasets import load_dataset
        for config in ["bio-retain-corpus", "cyber-retain-corpus", "cyber-forget-corpus"]:
            data = load_dataset("cais/wmdp-corpora", config, split="train")
            print(f"cached {config}: {len(data)} documents")


if __name__ == "__main__":
    main()
