#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import UNLEARNED_ROOT, unlearn_report_path  # noqa: E402
from src.unlearn.corpora import load_forget, load_retain  # noqa: E402
from src.unlearn.elm import register_checkpoint, run_elm  # noqa: E402
from src.unlearn.rmu import RMUConfig, run_rmu  # noqa: E402

OUT_ROOT = UNLEARNED_ROOT


def _write_report(out: Path, payload: dict) -> None:


    assert payload.get("base_model"), "a checkpoint without base_model provenance cannot be swept"
    unlearn_report_path(out).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-corpora", action="store_true",
                    help='Fetch corpora.')
    ap.add_argument("--method", required=True, choices=("rmu", "elm"))
    ap.add_argument("--name", default=None,
                    help='Name.')
    ap.add_argument("--model", default=None, help='Model.')
    ap.add_argument("--models-dir", default=os.environ.get("MODELS_DIR", "./models"))
    ap.add_argument("--concept", default="wmdp-cyber", choices=("wmdp-bio", "wmdp-cyber"))
    ap.add_argument("--from-checkpoint", default=None,
                    help='From checkpoint.')
    ap.add_argument("--forget-file", default=None,
                    help='Forget file.')
    ap.add_argument("--forget-limit", type=int, default=2000)
    ap.add_argument("--retain-limit", type=int, default=2000)

    ap.add_argument("--layer", default=None,
                    help='Layer.')
    ap.add_argument("--steering-coeff", type=float, default=6.5)
    ap.add_argument("--alpha", type=float, default=1200.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-batches", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42,
                    help='Seed.')
    ap.add_argument("--dtype", default="bfloat16",
                    help='Dtype.')
    ap.add_argument("--frozen-dtype", default=None,
                    help='Frozen dtype.')
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frozen-device", help='Frozen device.')
    ap.add_argument("--early-stop", action="store_true", help='Early stop.')
    ap.add_argument("--steering-ratio", type=float, help='Steering ratio.')
    args = ap.parse_args()

    if not args.name and not args.fetch_corpora:
        ap.error("--name is required (except with --fetch-corpora)")

    if args.fetch_corpora:


        f = load_forget(args.concept, limit=args.forget_limit, local_path=args.forget_file)
        r = load_retain(limit=args.retain_limit)
        print(f"cached: {len(f)} forget ({args.concept}) · {len(r)} retain (wikitext-103)")
        print("Corpora are cached. Run unlearn.py on an allocated GPU.")
        return 0

    if "/" in args.name or args.name in (".", ".."):
        raise SystemExit(f"--name must be a plain directory name, got {args.name!r} — a path here "
                         f"could place the checkpoint inside MODELS_DIR and overwrite a base model.")
    out = Path(OUT_ROOT) / args.name
    if out.exists() or out.is_symlink():
        raise SystemExit(f"{out} already exists — pick another --name, or remove it first. "
                         f"Overwriting silently would leave a results table pointing at a "
                         f"checkpoint that is no longer the one it measured.")

    if args.from_checkpoint:


        if not args.model:
            raise SystemExit(
                "--model is required with --from-checkpoint: name the base model this checkpoint was "
                "erased from (e.g. --model Meta-Llama-3-8B-Instruct). Without it the checkpoint has "
                "no provenance, `src/gen/thinkstate.py` cannot derive its reasoning capability, and "
                "there is no unedited row to compare its numbers against.")
        dst = register_checkpoint(args.from_checkpoint, args.name, OUT_ROOT)
        _write_report(dst, {"method": args.method, "base_model": args.model,
                            "model": str(Path(args.models_dir) / args.model),
                            "adopted_from": str(Path(args.from_checkpoint).resolve()),
                            "concept": args.concept})
        _print_next(args.name, args.concept)
        return 0

    if args.method == "elm":
        run_elm()
        return 1

    if args.layer is None:
        raise SystemExit("--layer is required for RMU. It is the layer whose activations are "
                         "steered, and it is architecture-specific: the paper uses 7 for Zephyr-7B "
                         "(32 layers). Check your model's depth before picking one.")
    if args.concept == "wmdp-bio" and not args.forget_file:
        raise SystemExit("wmdp-bio's forget corpus is GATED. Request access at "
                         "https://huggingface.co/datasets/cais/wmdp-corpora, then pass "
                         "--forget-file <the downloaded file>. Do not substitute other text: a "
                         "reproduction trained on a different corpus is not a reproduction.")
    if not args.model:
        raise SystemExit("--model is required when training")

    model_path = str(Path(args.models_dir) / args.model)
    if not Path(model_path).exists():
        raise SystemExit(f"no such model directory: {model_path}")

    if args.layer == "auto":
        from transformers import AutoConfig
        model_cfg = AutoConfig.from_pretrained(model_path)
        text_cfg = getattr(model_cfg, "text_config", model_cfg)
        args.layer = max(2, round(text_cfg.num_hidden_layers * 7 / 32))
    else:
        args.layer = int(args.layer)
    if min(args.batch_size, args.max_batches, args.lr, args.alpha, args.steering_coeff) <= 0:
        ap.error("batch size, steps, learning rate, alpha and steering coefficient must be positive")

    print(f"[{args.method}] {args.model} · {args.concept} · layer {args.layer}")
    forget = load_forget(args.concept, limit=args.forget_limit, local_path=args.forget_file)
    retain = load_retain(limit=args.retain_limit)
    print(f"  corpora: {len(forget)} forget · {len(retain)} retain")

    cfg = RMUConfig(layer=args.layer, steering_coeff=args.steering_coeff, alpha=args.alpha,
                    lr=args.lr, max_batches=args.max_batches, batch_size=args.batch_size,
                    seed=args.seed, early_stop=args.early_stop, steering_ratio=args.steering_ratio)
    report = run_rmu(model_path, out, concept=args.concept, forget=forget, retain=retain,
                     cfg=cfg, dtype=args.dtype, device=args.device,
                     frozen_dtype=args.frozen_dtype, frozen_device=args.frozen_device)

    _write_report(out, {"method": args.method, "base_model": args.model,
                        "forget_file": args.forget_file, **vars(report)})
    _print_next(args.name, args.concept)
    return 0


def _print_next(name: str, concept: str) -> None:


    print(f"""
NEXT — the gates, before any sweep (COMMANDS §8, gemma §8b):

  python scripts/diff_checkpoint.py {name}
  python -c "import json;r=json.load(open('models/unlearned/{name}/unlearn_report.json'));f=r['forget_losses'];print(f'forget {{f[0]:.4f}} -> {{f[-1]:.4f}} ({{(f[-1]/f[0]-1)*100:+.1f}}%)  fluency {{r.get(\\"fluency_unique_ratio\\")}}')"

Check matrix changes, then validation likelihood and chat parse/retain rates. Raw-text fluency
is not a Gemma gate. Measure unprompted with the wmdp8 instrument: rmu3 for Llama, rmu2 for E2B,
rmu4 for 31B. Trained target: {concept}; the other WMDP domain is a control.""")


if __name__ == "__main__":
    raise SystemExit(main())
