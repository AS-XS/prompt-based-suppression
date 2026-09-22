#!/usr/bin/env python


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.unlearn.corpora import load_forget  # noqa: E402
from src.unlearn.rmu import activation_index, decoder_layers_path, hidden_size_of  # noqa: E402


_PAPER_DEPTH = 7 / 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help='Model.')
    ap.add_argument("--models-dir", default="./models")
    ap.add_argument("--concept", default="wmdp-cyber", choices=("wmdp-bio", "wmdp-cyber"))
    ap.add_argument("--forget-file", default=None, help='Forget file.')
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help='Layers.')
    ap.add_argument("--batch", type=int, default=8, help='Batch.')
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tree", action="store_true",
                    help='Tree.')
    args = ap.parse_args()

    d = Path(args.model) if "/" in args.model else Path(args.models_dir) / args.model
    if not d.is_dir():
        raise SystemExit(f"no such model directory: {d}")

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(d)
    print(f"{args.model}\n  model_type {getattr(cfg, 'model_type', '?')}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "no GPU on this node — this is a forward pass over the model.\n"
            f"  qsub -N probe -v MODEL={args.model} scripts/qsub_probe.sh")

    model = AutoModelForCausalLM.from_pretrained(d, dtype=getattr(torch, args.dtype)).to(args.device)
    model.eval()

    if args.tree:
        for name, _ in list(model.named_children()):
            print(f"  model.{name}")
            for sub, _ in list(getattr(model, name).named_children()):
                print(f"    model.{name}.{sub}")
        return 0

    layers, path = decoder_layers_path(model)
    n = len(layers)
    hidden = hidden_size_of(cfg)
    print(f"  decoder blocks: {n} at `{path}`\n  hidden_size {hidden}")

    default = max(1, round(_PAPER_DEPTH * n))
    candidates = args.layers or sorted({max(2, default - 2), default, default + 2})
    print(f"  the paper's relative depth (7/32) on {n} layers -> layer {default}\n")

    missing = [i for i in range(n) if not hasattr(getattr(layers[i], "mlp", None), "down_proj")]
    if missing:
        print(f"  ⚠️ {len(missing)} layer(s) have no `mlp.down_proj` — RMU is DEFINED as editing that "
              f"matrix on layers ℓ-2..ℓ, so it cannot be reproduced here as published.\n"
              f"     first few: {missing[:6]}")
        return 1
    print(f"  ✓ every layer has mlp.down_proj")

    tok = AutoTokenizer.from_pretrained(d)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = load_forget(args.concept, limit=args.batch, local_path=args.forget_file)[:args.batch]
    batch = tok([t[:2000] for t in texts], return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(args.device)
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)

    print(f"\n  activation scale on {len(texts)} {args.concept} documents "
          f"(hidden_states[ℓ+1] = the OUTPUT of layers[ℓ]):")
    print(f"    {'layer':>6} {'mean ||h||':>12} {'RMS ||h||':>12}   {'suggested c (1.1x RMS)':>24}")
    for ell in candidates:
        if ell >= n:
            continue
        h = out.hidden_states[activation_index(ell, len(out.hidden_states))].float()
        mean = float(h.norm(dim=-1).mean())
        rms = float(h.pow(2).mean().sqrt() * hidden ** 0.5)
        star = "  <- paper depth" if ell == default else ""
        print(f"    {ell:6d} {mean:12.2f} {rms:12.2f}   {1.1 * rms:24.1f}{star}")

    print(f"\n  Llama-3-8B for reference: layer 7, RMS 17.68, worked at c=20 (1.13x) with alpha=300.")
    print(f"  Start there; alpha=1200 removed nothing and alpha<=100 destroyed the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
