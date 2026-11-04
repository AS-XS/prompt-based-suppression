#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import UNLEARNED_ROOT, unlearn_report_path  # noqa: E402
from src.unlearn.elm import layer_window  # noqa: E402
from src.unlearn.elm_upstream import REVISION, SOURCE_SHA256, adapted_tree  # noqa: E402
from src.unlearn.elm_resume import Recovery, TrainingPaused, lock_work  # noqa: E402
from src.unlearn.elm_targets import get_elm_peft_model  # noqa: E402
from src.unlearn.rmu import _fingerprint, _git_commit, complete_checkpoint  # noqa: E402
from src.unlearn.tensors import carry_over_missing  # noqa: E402


def main():
    started = time.monotonic()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--name", required=True)
    ap.add_argument("--concept", required=True, choices=["wmdp-bio", "wmdp-cyber"])
    ap.add_argument("--forget-file", type=Path)
    ap.add_argument("--eta", type=int, default=500)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--layers", default="4,8",
                    help='Layers.')
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--source-dir", type=Path, default=Path("data/external/elm"))
    ap.add_argument("--resume", action="store_true", help='Resume.')
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--max-seconds", type=int, default=37800,
                    help='Max seconds.')
    args = ap.parse_args()
    if args.mixed_precision:
        args.dtype = "bfloat16"
    if args.checkpoint_every <= 0 or args.max_seconds < 0:
        ap.error("checkpoint-every must be positive and max-seconds nonnegative")
    if Path(args.name).name != args.name or args.name in {".", ".."}:
        ap.error("name must be a plain directory name")
    if args.concept == "wmdp-bio" and (not args.forget_file or not args.forget_file.is_file()):
        ap.error("bio needs --forget-file pointing to the authorized JSONL corpus")
    if args.samples <= 0 or args.samples % 4:
        ap.error("samples must be positive and divisible by the authors' accumulation factor 4")
    base = (Path(args.models_dir) / args.model).resolve()
    out = (UNLEARNED_ROOT / args.name).resolve()
    work = (Path("results/elm-training") / args.name).resolve()
    if out.exists() and unlearn_report_path(out).exists():
        ap.error("completed checkpoint already exists; do not resubmit")
    if out.exists() or work.exists():
        if not args.resume or not (work / "resume.pt").is_file():
            ap.error("existing run has no resumable state; choose a fresh name")
    _training_lock = lock_work(work)
    if unlearn_report_path(out).exists():
        ap.error("completed checkpoint already exists; do not resubmit")
    if base == out or base in out.parents or out in base.parents:
        ap.error("base and output directories must be disjoint")
    source = (args.source_dir / "erase.py").read_text()
    tree = adapted_tree(source, recovery=True)
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    if not torch.cuda.is_available():
        ap.error("ELM must run on a CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(base)
    depth = getattr(config, "text_config", config).num_hidden_layers
    try:
        start, end = layer_window(args.layers, depth)
    except ValueError as e:
        ap.error(str(e))
    if args.parallel and torch.cuda.device_count() < 2:
        ap.error("--parallel requires at least two visible GPUs")
    before = _fingerprint(base)

    def load_model(model_id, dtype):
        options = dict(dtype=dtype)
        if args.parallel:
            options.update(device_map="balanced" if args.mixed_precision else "auto",
                max_memory={i: "40GiB" if args.mixed_precision else "65GiB" for i in range(2)})
        model = AutoModelForCausalLM.from_pretrained(model_id, **options)
        if args.parallel:
            if any(str(d) in {"cpu", "disk"} for d in model.hf_device_map.values()):
                raise RuntimeError("model did not fit on two GPUs; CPU/disk training offload refused")
        else:
            model.to("cuda:0")
        return model

    def adapter_model(model, config, peft_factory):
        adapted = get_elm_peft_model(model, config, peft_factory)
        if args.mixed_precision:
            from src.unlearn.precision import promote_trainable
            promote_trainable(adapted)
        return adapted

    namespace = {"__name__": "elm_pinned", "_load_model": load_model,
                 "_get_elm_peft_model": adapter_model}
    exec(compile(tree, str(args.source_dir / "erase.py"), "exec"), namespace)
    original_prepare = namespace["prepare_prompts"]
    corpus_manifest = {}

    def prepare(*a, **kw):
        kw["rmu_keywords_path"] = str((args.source_dir / "wmdp-keywords.json").resolve())
        if args.forget_file:
            kw["bio_corpus_path"] = str(args.forget_file.resolve())
        prepared = original_prepare(*a, **kw)
        for kind, texts in zip(["forget", "retain"], prepared[:2]):
            corpus_manifest[kind] = {
                str(domain): dict(count=len(rows), sha256=hashlib.sha256(
                    json.dumps(rows, ensure_ascii=False).encode()).hexdigest())
                for domain, rows in texts.items()}
        return prepared

    namespace["prepare_prompts"] = prepare
    options = SimpleNamespace(
        save_path=str(work.parent), experiment_name=args.name, model_id=str(base),
        device="cuda:0", dtype=getattr(torch, args.dtype), layers_to_train=f"{start},{end}",
        lora_rank=args.rank, lora_alpha=16, train_method="mlp-attn", action="erase",
        eta=args.eta, topk=50, temperature=1.2, use_erase_soft_loss="True",
        use_retain_soft_loss="False", lr=args.lr, loss="cross", verbose="False",
        num_samples=args.samples, dataset_idx="0" if args.concept == "wmdp-bio" else "1",
        min_len=50, max_len=700, erase_loss_scale=1, retain_loss_scale=1,
        consistence_loss_scale=1, grad_accumulation_steps=4, wandb_log=0,
        pregenerated_consistency_path=None, consistence_type="normal", save_every=50000)
    provenance = dict(method="elm", base_model=args.model, model=str(base), concept=args.concept,
                      git_commit=_git_commit(), upstream_revision=REVISION,
                      upstream_source_sha256=SOURCE_SHA256, seed=args.seed,
                      config=vars(options), parallel=args.parallel,
                      keywords_sha256=hashlib.sha256(
                          (args.source_dir / "wmdp-keywords.json").read_bytes()).hexdigest(),
                      forget_file=str(args.forget_file) if args.forget_file else None)
    identity = json.loads(json.dumps(dict(config=vars(options), seed=args.seed,
        parallel=args.parallel, base_fingerprint=before, corpora=corpus_manifest,
        source=SOURCE_SHA256, torch_version=str(torch.__version__),
        keywords=provenance["keywords_sha256"]), default=str))
    if args.mixed_precision:
        precision = dict(backbone='bfloat16', trainable='float32', placement='balanced-40GiB')
        identity['precision'] = precision
        provenance['precision'] = precision

    identity["corpora"] = corpus_manifest
    namespace["_recovery"] = Recovery(work, identity, resume=args.resume,
        every=args.checkpoint_every, seconds=args.max_seconds, started=started)
    provenance["recovery"] = dict(resume=args.resume, checkpoint_every=args.checkpoint_every,
                                  max_seconds=args.max_seconds)
    if not (work / "request.json").exists():
        (work / "request.json").write_text(json.dumps(provenance, indent=2, default=str))
    with (work / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps(provenance, default=str) + "\n")
    try:
        model, tok, adapter, losses = namespace["train_elm"](options)
    except TrainingPaused as paused:
        print(str(paused), flush=True)
        return
    (work / "losses.json").write_text(json.dumps(losses))
    merged = model.merge_and_unload(safe_merge=True)

    merged.to(dtype=torch.bfloat16)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, max_shard_size="5GB")
    tok.save_pretrained(out)
    complete_checkpoint(base, out)
    carry_over_missing(base, out)
    if _fingerprint(base) != before:
        raise RuntimeError("base model changed during ELM; checkpoint not certified")
    provenance.update(adapter=adapter, loss_file=str(work / "losses.json"),
                      saved_dtype="bfloat16", corpora=corpus_manifest)
    unlearn_report_path(out).write_text(json.dumps(provenance, indent=2, default=str))
    print(f"ELM checkpoint saved: {out}. Evaluate validation likelihood before a full sweep.")


if __name__ == "__main__":
    main()
