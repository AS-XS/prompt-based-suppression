import json
import time
from pathlib import Path

from src.paths import UNLEARNED_ROOT, unlearn_report_path, r2mu_training_dir
from src.unlearn.activations import block_output
from src.unlearn.elm_resume import Recovery, TrainingPaused, lock_work
from src.unlearn.r2mu_data import AUTHOR_REVISION, digest, load_bundle, read_rows
from src.unlearn.rmu import (_decoder_layers, _fingerprint, _git_commit,
                            complete_checkpoint, hidden_size_of)
from src.unlearn.tensors import carry_over_missing


def masked_mse(actual, target, mask):
    error = (actual.float() - target.float()).square().mean(dim=-1)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.unlearn.corpora import load_retain

    started = time.monotonic()
    src = (Path(args.models_dir) / args.model).resolve()
    dst = (UNLEARNED_ROOT / args.name).resolve()
    work = r2mu_training_dir(args.name)
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('source/output overlap')
    if unlearn_report_path(dst).exists():
        raise ValueError('checkpoint already complete')
    if (work.exists() or dst.exists()) and not (args.resume and (work / 'resume.pt').is_file()):
        raise ValueError('existing run without resumable state; choose a fresh name')
    manifest, rows, traces = load_bundle(args.data, args.model, args.concept)
    limo = read_rows(args.limo_file)
    if not limo or any(not isinstance(r.get('solution'), str) or not r['solution'].strip() for r in limo):
        raise ValueError('LIMO solutions missing; run prepare_r2mu.py --fetch-limo')
    if len(rows) < args.steps * args.batch_size:
        raise ValueError('not enough prepared traces for requested steps')
    retain = load_retain(limit=args.steps * args.batch_size)
    if len(retain) < args.steps * args.batch_size:
        raise ValueError('not enough retain corpus')
    if args.layer == 'auto':
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(src)
        args.layer = max(2, round(getattr(config, 'text_config', config).num_hidden_layers * 7 / 32))
    else:
        args.layer = int(args.layer)
    mixed = getattr(args, 'mixed_precision', False)
    ratio = getattr(args, 'steering_ratio', None)
    before = _fingerprint(src)
    if digest(before) != digest(manifest['base_fingerprint']):
        raise ValueError('base checkpoint differs from trace-generation model')
    identity = json.loads(json.dumps(dict(model=args.model, concept=args.concept,
        layer=args.layer, steps=args.steps, batch_size=args.batch_size, coeff=args.coeff,
        gamma=args.gamma, steering_ratio=ratio, trace_weight=args.trace_weight, cot_weight=args.cot_weight,
        lr=args.lr, seed=args.seed, data=digest(rows), limo=digest(limo), retain=digest(retain),
        base=before, dtype='float32', frozen_dtype=args.frozen_dtype,
        device=args.device, frozen_device=args.frozen_device, version=1), default=str))
    if mixed:
        identity.update(dtype='bfloat16', trainable_dtype='float32', mixed_precision=True)
    _lock = lock_work(work)
    if unlearn_report_path(dst).exists():
        raise ValueError('checkpoint already complete')
    import random
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(src)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    updated = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16 if mixed else torch.float32).to(args.device)
    frozen = AutoModelForCausalLM.from_pretrained(
        src, dtype=getattr(torch, args.frozen_dtype)).to(args.frozen_device)
    frozen.requires_grad_(False); frozen.eval()
    updated.requires_grad_(False)
    blocks = _decoder_layers(updated)
    if not 2 <= args.layer < len(blocks):
        raise ValueError('layer must have two preceding decoder blocks')
    for i in range(args.layer-2, args.layer+1):
        blocks[i].mlp.down_proj.weight.requires_grad_(True)
    if mixed:
        from src.unlearn.precision import promote_trainable
        promote_trainable(updated)
    updated.train()
    params = [p for p in updated.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    rng = torch.Generator(device=args.device).manual_seed(args.seed)
    control = torch.rand(1, 1, hidden_size_of(updated.config), device=args.device, generator=rng)
    coeff = args.coeff
    if ratio is not None:
        updated.eval()
        with torch.no_grad():
            batch = tokenizer([r['forget'] for r in rows[:args.batch_size]],
                return_tensors='pt', padding=True, truncation=True, max_length=512).to(args.device)
            h = block_output(updated, blocks[args.layer], batch).float()
            mask = batch['attention_mask']
            coeff = ratio * float(((h.square().sum(-1) * mask).sum() / mask.sum()).sqrt())
        del h, batch
        updated.train()
    control = control / control.norm() * coeff
    recovery = Recovery(work, identity, resume=args.resume, every=10,
                        seconds=args.max_seconds, started=started, label='R2MU')
    start, _, _, losses = recovery.restore(updated, optimizer)
    request = dict(method='r2mu', adaptation='Gemma Eq.6 pilot', base_model=args.model,
                   model=str(src), concept=args.concept, config=identity, git_commit=_git_commit(),
                   author_revision=AUTHOR_REVISION, preparation=manifest,
                   resolved_coeff=coeff, selected_trace_indices=[r['index'] for r in rows[:args.steps*args.batch_size]])
    with (work / 'attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(request) + '\n')

    def term(texts, preserve):
        batch = tokenizer(texts, return_tensors='pt', padding=True, truncation=True,
                          max_length=512).to(args.device)
        actual = block_output(updated, blocks[args.layer], batch)
        if preserve:
            with torch.no_grad():
                other = {k: v.to(args.frozen_device) for k, v in batch.items()}
                target = block_output(frozen, _decoder_layers(frozen)[args.layer], other)
            target = target.to(actual.device)
        else:
            target = control
        return masked_mse(actual, target, batch['attention_mask'])

    try:
        for step in range(start, args.steps):
            a, b = step * args.batch_size, (step+1) * args.batch_size
            batches = (
                ('forget', [r['forget'] for r in rows[a:b]], False, 1.0),
                ('retain', retain[a:b], True, args.gamma),
                ('trace', traces[a:b], False, args.trace_weight),
                ('cot', [limo[i % len(limo)]['solution'] for i in range(a,b)], True, args.cot_weight),
            )
            optimizer.zero_grad()
            total = 0.0
            for key, texts, preserve, weight in batches:
                value = 0.0
                if weight:
                    loss = term(texts, preserve)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Non-finite {key} loss at step {step+1}; no update applied')
                    (loss * weight).backward()
                    value = float(loss.detach())
                    del loss
                losses.setdefault(key, []).append(value)
                total += weight * value
            optimizer.step(); optimizer.zero_grad()
            losses.setdefault('total', []).append(total)
            print(f'R2MU step {step+1}/{args.steps}: ' + ' '.join(
                f'{key}={values[-1]:.6g}' for key,values in losses.items()), flush=True)
            recovery.boundary(updated, optimizer, step+1, step, {}, losses, 1, args.steps)
    except TrainingPaused as paused:
        print(str(paused), flush=True); return
    del frozen, optimizer
    torch.cuda.empty_cache()
    dst.mkdir(parents=True, exist_ok=True)
    updated.to(dtype=torch.bfloat16).save_pretrained(dst, max_shard_size='5GB')
    tokenizer.save_pretrained(dst)
    complete_checkpoint(src, dst); carry_over_missing(src, dst)
    if _fingerprint(src) != before:
        raise RuntimeError('source model changed; checkpoint not certified')
    request['losses'] = losses
    request['saved_dtype'] = 'bfloat16'
    temp = dst / 'unlearn_report.tmp'
    temp.write_text(json.dumps(request, indent=2))
    temp.replace(unlearn_report_path(dst))
    print(f'R2MU checkpoint saved: {dst}', flush=True)
