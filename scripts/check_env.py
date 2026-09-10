#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gen.env import make_llm, setup_vllm_env  # noqa: E402
from src.prompts import build_messages  # noqa: E402


_PREFERRED = ["Llama-3.2-1B-Instruct", "Qwen2.5-1.5B-Instruct", "Qwen3-1.7B", "Llama-3.2-3B-Instruct"]


def pick_model(models_dir: Path) -> Path:
    for name in _PREFERRED:
        p = models_dir / name
        if p.is_dir() and any(p.iterdir()):
            return p

    for p in sorted(models_dir.iterdir()):
        if p.is_dir() and any(p.iterdir()):
            return p
    raise SystemExit(f"No model dirs found under {models_dir} — set MODELS_DIR or --model.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help='Model.')
    ap.add_argument("--concept", type=str, default="biology")
    args = ap.parse_args()

    setup_vllm_env()


    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"GPU(s): {gpu.replace(chr(10), ', ') if gpu else 'NONE — on a GPU node? (scripts/gpu.sh)'}")

    model_path = Path(args.model) if args.model else pick_model(
        Path(os.environ.get("MODELS_DIR", "./models"))
    )
    print(f"loading (vLLM): {model_path}")

    from vllm import SamplingParams  # noqa: E402
    llm = make_llm(str(model_path), dtype="auto", gpu_memory_utilization=0.85, max_model_len=8192)

    messages = build_messages("world-frame", args.concept,
                              f"Give a one-sentence overview of {args.concept}.")
    out = llm.chat([messages], SamplingParams(temperature=0.0, max_tokens=128))
    text = out[0].outputs[0].text.strip()

    print("\n=== generation OK ===")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
