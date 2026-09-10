#!/usr/bin/env python


from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gen.env import make_llm, setup_vllm_env  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "models.txt"
SENTINEL = "RESULT::"


def roster() -> list[str]:
    return [ln.split("#", 1)[0].strip() for ln in MANIFEST.read_text().splitlines()
            if ln.split("#", 1)[0].strip()]


def run_one(path: str, max_tokens: int) -> int:


    try:
        setup_vllm_env()
        from vllm import SamplingParams

        t0 = time.time()

        llm = make_llm(path, dtype="auto", gpu_memory_utilization=0.85, enforce_eager=True,
                       max_model_len=8192)
        load_s = time.time() - t0

        t1 = time.time()
        out = llm.generate(["Give a one-sentence overview of biology."],
                           SamplingParams(temperature=0.0, max_tokens=max_tokens))
        gen_s = max(time.time() - t1, 1e-6)
        o = out[0].outputs[0]
        preview = o.text.strip().replace("\n", " ")[:50]
        print(f"{SENTINEL}OK load={load_s:.0f}s gen={gen_s:.1f}s "
              f"{len(o.token_ids) / gen_s:.1f}tok/s :: {preview}")
        return 0
    except Exception as e:  # noqa: BLE001 — any failure is a per-model FAIL, not a crash
        print(f"{SENTINEL}FAIL {type(e).__name__}: {str(e)[:90]}")
        return 1


def _result(text: str) -> str:
    for line in reversed(text.splitlines()):
        if SENTINEL in line:
            return line.split(SENTINEL, 1)[1]
    return ""


def _root_cause(text: str) -> str:

    hits = [ln.split("] ")[-1].strip() for ln in text.splitlines()
            if ("Error:" in ln or "Exception:" in ln) and "See root cause above" not in ln]
    return hits[-1][:140] if hits else ""


def _has_gpu() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _kill_group(proc: "subprocess.Popen") -> None:

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help='Names.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--timeout", type=int, default=420, help='Timeout.')
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--one", help='One.')
    args = ap.parse_args()

    if args.one:
        return run_one(args.one, args.max_tokens)

    if not _has_gpu():
        print("No GPU detected (nvidia-smi failed) — run on a GPU node: ./scripts/gpu.sh")
        return 1

    busy = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if busy:
        print("WARNING: the GPU is already occupied — checks may fail with 'device busy'.\n"
              f"  running: {busy.replace(chr(10), '; ')}\n"
              "  clear with: nvidia-smi --query-compute-apps=pid --format=csv,noheader | "
              "xargs -r kill -9   (or use a fresh GPU)\n")

    root = Path(args.models_dir)
    wanted = set(args.names) or None
    dirs = [d for d in (root / r.split("/")[-1] for r in roster())
            if d.is_dir() and (wanted is None or d.name in wanted)]
    print(f"vLLM check: {len(dirs)} model(s) in {root}, timeout {args.timeout}s each\n")

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    bad = 0
    for d in dirs:
        cmd = [sys.executable, __file__, "--one", str(d), "--max-tokens", str(args.max_tokens)]
        cause = ""

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=args.timeout)
            line = _result(out) or "engine failed"
            status = "OK" if (proc.returncode == 0 and line.startswith("OK")) else "FAIL"
            cause = _root_cause(out) if status == "FAIL" else ""
        except subprocess.TimeoutExpired:
            status, line = "FAIL", f"TIMEOUT >{args.timeout}s (hang or CPU fallback)"
        finally:
            _kill_group(proc)
        bad += status != "OK"
        print(f"  [{status:4}] {d.name:34} {line}")
        if cause:
            print(f"           ↳ {cause}")

    print(f"\n{'ALL GOOD' if not bad else f'{bad} model(s) FAILED'} ({len(dirs)} checked)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
