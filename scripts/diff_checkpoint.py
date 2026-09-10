#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import UNLEARNED_ROOT, unlearn_report_path  # noqa: E402


_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
           "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8", "U8": "uint8",
           "BOOL": "bool"}
_CHUNK = 8 << 20


def updated_param_names(report: dict) -> list[str]:


    cfg = report.get("config") or {}
    layers = cfg.get("layers_to_update")
    if not layers:
        layer = cfg.get("layer")
        if layer is None:
            raise SystemExit("the report records neither `layers_to_update` nor `layer`, so there "
                             "is no claim to check.")
        layers = list(range(max(layer - 2, 0), layer + 1))
    return [f"model.layers.{i}.mlp.down_proj.weight" for i in layers]


def read_header(path: Path) -> tuple[dict, int]:


    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def shard_index(model_dir: Path) -> dict[str, str]:

    idx = model_dir / "model.safetensors.index.json"
    if idx.is_file():
        return json.loads(idx.read_text(encoding="utf-8"))["weight_map"]
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise SystemExit(f"{model_dir} has no model.safetensors or index — is it a HF checkpoint?")
    header, _ = read_header(single)
    return {k: single.name for k in header if k != "__metadata__"}


def locate(model_dir: Path, index: dict[str, str], name: str) -> tuple[Path, str, list, int, int]:

    path = model_dir / index[name]
    header, base = read_header(path)
    info = header[name]
    start, end = info["data_offsets"]
    return path, info["dtype"], info["shape"], base + start, base + end


def _read(loc) -> "object":
    import torch

    path, dtype, shape, start, end = loc
    if dtype not in _DTYPES:
        raise SystemExit(f"unsupported dtype {dtype} in {path}")
    with path.open("rb") as f:
        f.seek(start)
        buf = bytearray(f.read(end - start))
    return torch.frombuffer(buf, dtype=getattr(torch, _DTYPES[dtype])).reshape(shape)


def identical(a, b) -> bool:

    (pa, _, _, sa, ea), (pb, _, _, sb, eb) = a, b
    if ea - sa != eb - sb:
        return False
    with pa.open("rb") as fa, pb.open("rb") as fb:
        fa.seek(sa); fb.seek(sb)
        left = ea - sa
        while left > 0:
            n = min(_CHUNK, left)
            if fa.read(n) != fb.read(n):
                return False
            left -= n
    return True


def stats(a, b, chunk: int = 512) -> tuple[float, float, float]:


    import torch

    mx = total = base_total = 0.0
    n = 0
    for i in range(0, a.shape[0], chunk):
        x = a[i:i + chunk].to(torch.float32)
        y = b[i:i + chunk].to(torch.float32)
        d = (x - y).abs()
        mx = max(mx, float(d.max()))
        total += float(d.sum())
        base_total += float(y.abs().sum())
        n += d.numel()
    mean = total / n if n else 0.0
    return mx, mean, mean / (base_total / n) if base_total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help='Name.')
    ap.add_argument("--base", default=None,
                    help='Base.')
    ap.add_argument("--others", type=int, default=12,
                    help='Others.')
    args = ap.parse_args()

    ckpt = UNLEARNED_ROOT / args.name
    report = json.loads(unlearn_report_path(ckpt).read_text(encoding="utf-8"))
    base = Path(args.base or report.get("model") or "")
    if not base.is_dir():
        raise SystemExit(f"base model not found at {base} — pass --base explicitly.")

    targets = updated_param_names(report)
    ci, bi = shard_index(ckpt), shard_index(base)
    missing = [t for t in targets if t not in ci or t not in bi]
    if missing:
        raise SystemExit(f"parameters named in the report are absent from the weights: {missing}")

    cfg = report.get("config") or {}
    print(f"{args.name}  vs  {base}")
    print(f"method {report.get('method')} · base_model {report.get('base_model')} · "
          f"dtype {cfg.get('dtype', 'bfloat16')}")


    print(f"c={cfg.get('steering_coeff')} alpha={cfg.get('alpha')} lr={cfg.get('lr')} "
          f"layers={cfg.get('layers_to_update')} · commit {report.get('git_commit') or 'UNRECORDED'}"
          f"{'  <- predates the git_commit field' if not report.get('git_commit') else ''}\n")
    print(f"{'parameter':52} {'max |Δ|':>12} {'mean |Δ|':>12} {'rel':>9}")
    print("-" * 90)
    changed = 0
    for t in targets:
        mx, mean, rel = stats(_read(locate(ckpt, ci, t)), _read(locate(base, bi, t)))
        changed += mx > 0
        print(f"{t:52} {mx:12.3e} {mean:12.3e} {rel:8.3%}"
              f"{'   <- UNCHANGED' if mx == 0 else ''}")

    others = [k for k in sorted(ci) if k not in targets and k in bi]
    if args.others and others:
        sample = others if args.others < 0 else others[:: max(1, len(others) // args.others)]
        touched = [k for k in sample
                   if not identical(locate(ckpt, ci, k), locate(base, bi, k))]
        print(f"\nchecked {len(sample)} of {len(others)} other parameters — "
              f"{'ALL UNCHANGED' if not touched else f'CHANGED: {touched[:5]}'}")
        if touched:
            print("  ⚠️ the edit was NOT narrow. RMU's claim is that only down_proj on the chosen "
                  "layers moves; anything else here means a different method wearing its name.")

    print()
    if changed == 0:
        print("VERDICT: the weights are BIT-IDENTICAL to the base model. This checkpoint cannot")
        print("  score differently, so any accuracy comparison against it measures nothing. The")
        print("  training loop ran but no update survived — note that a pure-bf16 AdamW at lr=5e-5")
        print("  can round every individual update to zero without erroring (--dtype float32).")
        return 1
    print(f"VERDICT: {changed}/{len(targets)} claimed parameters changed. The edit is REAL, so a")
    print("  null accuracy result is a finding about RMU on this model, not a failed run. Read the")
    print("  `rel` column: it is how far the weights moved relative to their own magnitude.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
