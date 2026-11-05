#!/usr/bin/env python3

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.r2mu_author import DEFAULT_SOURCE, HASHES, REVISION, load_author, sha
from src.unlearn.r2mu_author_data import training_data
from src.unlearn.rmu import _decoder_layers, _fingerprint, _git_commit, complete_checkpoint
from src.unlearn.tensors import carry_over_missing
from src.unlearn.elm_resume import lock_work

MODELS = {'ds8': 'DeepSeek-R1-Distill-Llama-8B', 'g12': 'gemma-4-12B-it'}


class LocalLogger:
    def __init__(self, path):
        self.path = path
        self.steps = 0
    def init(self, **kwargs):
        return self
    def log(self, values):
        if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite author training metric; final export refused')
        with self.path.open('a') as f:
            f.write(json.dumps(values)+'\n')
        self.steps += 1
        print(f'R2MU author step {self.steps}/500 loss={values["loss"]:.6g}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=MODELS, required=True)
    ap.add_argument('--models-dir', type=Path, default=Path('models'))
    ap.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    ap.add_argument('--forget-file', type=Path, default=Path('data/raw/bio-forget-corpus.jsonl'))
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--name', help='Name.')
    args = ap.parse_args()
    gemma = args.variant == 'g12'
    if gemma != bool(args.bundle):
        ap.error('g12 requires --bundle; ds8 uses the authors released traces')
    name = args.name or f'r2mu-author-{args.variant}-bio'
    if Path(name).name != name or name in {'.', '..'}:
        ap.error('name must be a single directory name')
    base = (args.models_dir / MODELS[args.variant]).resolve()
    dst = Path('models/unlearned') / name
    work = Path('results/r2mu-author-training') / name
    if dst.exists() or (work.exists() and any(work.iterdir())):
        ap.error(f'Existing run {name}; this author loop has no optimizer resume. Inspect before retrying.')
    if not (base/'config.json').is_file():
        ap.error(f'Missing base model: {base}')
    before = _fingerprint(base)
    lock = lock_work(work)
    logger = LocalLogger(work/'losses.jsonl')
    namespace = load_author(args.source, logger, gemma=gemma)
    author_forward = namespace['forward_with_cache']
    def placed_forward(model, inputs, module, no_grad=True):
        return author_forward(model, {k:v.to(model.device) for k,v in inputs.items()},
                              module, no_grad=no_grad)
    namespace['forward_with_cache'] = placed_forward
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 2:
        ap.error('Requires exactly two assigned GPUs: updated on cuda:0, reference on cuda:1')
    random.seed(0)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)


    frozen = AutoModelForCausalLM.from_pretrained(base, dtype='auto', device_map={'': 1})
    updated = AutoModelForCausalLM.from_pretrained(base, dtype='auto', device_map={'': 0})
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.mask_token_id = tokenizer.sep_token_id = tokenizer.cls_token_id = tokenizer.eos_token_id
    layer = max(2, round(len(_decoder_layers(updated))*7/32)) if gemma else 7
    settings = SimpleNamespace(output_dir=str(work), wandb_group='author-reference',
        layer_ids=list(range(layer-2, layer+1)), param_ids=[6], layer_id=layer,
        module_str='_blocks({model_name})[{layer_id}]' if gemma else '{model_name}.model.layers[{layer_id}]',
        lr=7.5e-5, alpha=[1.4,1.4], steering_coeff_list=[6.5,6.5],
        assist_loss=1., max_num_batches=500, batch_size=4, seed=42)
    if gemma:
        namespace['get_params'] = lambda model, layers, ids: [
            _decoder_layers(model)[i].mlp.down_proj.weight for i in layers]
    params = namespace['get_params'](updated, settings.layer_ids, settings.param_ids)
    selected = {id(p) for p in params}
    names = [n for n,p in updated.named_parameters() if id(p) in selected]
    if len(names) != 3 or any(not n.endswith('mlp.down_proj.weight') for n in names):
        raise ValueError(f'Unexpected upstream parameter selection: {names}')
    initial = [p.detach().cpu().clone() for p in params]
    forget, retain, traces, data = training_data(args.source, args.forget_file,
        args.bundle, MODELS[args.variant], base, work)
    extent = namespace['get_dataset_with_generated'](str(args.forget_file), str(traces), tokenizer,
                                                     max_gen_tokens=100, batch_size=4)
    user, assistant, reasoning = namespace['get_limo_dataset'](tokenizer, batch_size=4)
    from datasets import load_dataset
    from importlib.metadata import version
    data['limo_fingerprint'] = load_dataset('GAIR/LIMO', split='train')._fingerprint
    if min(len(forget[0]),len(retain[0]),len(extent[0]),len(assistant[0])) < 500:
        raise ValueError('Insufficient batches: refuse a silently shortened reference run')
    report = dict(method='r2mu-author', base_model=MODELS[args.variant], model=str(base),
        concept='wmdp-bio', author_revision=REVISION, upstream_hashes=HASHES,
        git_commit=_git_commit(), config=vars(settings), data=data, selected_parameters=names,
        dtype=str(updated.dtype), source_fingerprint=before,
        package_versions={p:version(p) for p in ('torch','transformers','datasets')},
        adaptations=['local corpus/cache paths', 'fixed undefined LIMO assistant_msg',
            'local JSONL metrics', 'one model copy per GPU', 'evaluation separate'] +
            (['native Gemma delimiters/traces', 'depth-scaled decoder layer', 'named down projections',
              'nested hidden size'] if gemma else []))
    (work/'request.json').write_text(json.dumps(report, indent=2))

    final = Path(namespace['run_rmu'](updated, frozen, tokenizer, forget, retain, extent,
                                      user, assistant, reasoning, settings))
    if logger.steps != 500:
        raise ValueError(f'Expected 500 updates, observed {logger.steps}')
    report['changed_elements'] = {n:int((p.detach().cpu()!=old).sum())
                                 for n,p,old in zip(names,params,initial)}
    report['completed_steps'] = logger.steps
    if _fingerprint(base) != before:
        raise RuntimeError('Base model files changed during training')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(final), str(dst))
    report['copied_serving_files'] = complete_checkpoint(base, dst)
    report['carried_tensors'] = carry_over_missing(base, dst)
    report['losses_sha256'] = sha(work/'losses.jsonl')
    (dst/'unlearn_report.json').write_text(json.dumps(report, indent=2))
    print(f'R2MU author checkpoint saved: {dst}; changed elements={report["changed_elements"]}')
    if not any(report['changed_elements'].values()):
        print('WARNING: no selected weights changed at stored precision; do not call this successful unlearning')


if __name__ == '__main__':
    main()
