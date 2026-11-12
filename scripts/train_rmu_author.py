#!/usr/bin/env python3

import argparse
import builtins
import json
import math
import random
import re
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.unlearn.rmu_author import ROOT, MODELS, FILES, REVISION, load_author
from src.unlearn.rmu_author_data import load_bundle
from src.unlearn.rmu import _decoder_layers, _fingerprint, _git_commit, complete_checkpoint
from src.unlearn.tensors import carry_over_missing
from src.unlearn.elm_resume import lock_work


class Metrics:
    def __init__(self, work):
        self.work, self.steps = work, 0

    def __call__(self, *values, **kwargs):
        text = ' '.join(map(str, values))
        builtins.print(*values, **kwargs, flush=True)
        with (self.work/'training.log').open('a') as f: f.write(text+'\n')
        if text.startswith('loss: '):
            values = {k:float(v) for k,v in re.findall(
                r'(loss|unlearn_loss|retain_loss|param_change): ([^ |]+)', text)}
            if len(values) != 4 or not all(math.isfinite(v) for v in values.values()):
                raise ValueError(f'Invalid author metric: {text}')
            self.steps += 1
            with (self.work/'losses.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=self.steps, **values))+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=MODELS, required=True)
    ap.add_argument('--models-dir', type=Path, default=Path('models'))
    ap.add_argument('--source', type=Path, default=ROOT)
    ap.add_argument('--name', help='Name.')
    args = ap.parse_args()
    name = args.name or f'rmu-author-{args.variant}-joint'
    if Path(name).name != name or name in {'.', '..'}: ap.error('name must be a plain directory name')
    base = (args.models_dir/MODELS[args.variant]).resolve()
    dst = Path('models/unlearned')/name
    work = Path('results/rmu-author-training')/name
    if dst.exists() or (work.exists() and any(work.iterdir())):
        ap.error(f'Existing run {name}; inspect it before using a fresh --name (no resume)')
    if not (base/'config.json').is_file(): ap.error(f'Missing base model {base}')
    forget, retain, data = load_bundle(args.source)
    gemma = args.variant == 'g12'
    namespace = load_author(args.source, gemma=gemma)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 2:
        ap.error('Requires two assigned GPUs: updated cuda:0, frozen reference cuda:1')
    before = _fingerprint(base)
    lock = lock_work(work)
    metrics = Metrics(work)
    namespace['print'] = metrics
    random.seed(0)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)
    frozen = AutoModelForCausalLM.from_pretrained(base, dtype='auto', device_map={'': 1})
    updated = AutoModelForCausalLM.from_pretrained(base, dtype='auto', device_map={'': 0})
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.mask_token_id = tokenizer.sep_token_id = tokenizer.cls_token_id = tokenizer.eos_token_id
    layer = max(2, round(len(_decoder_layers(updated))*7/32)) if gemma else 7
    settings = SimpleNamespace(model_name_or_path=str(base), output_dir=str(work/'checkpoint'),
        layer_ids=list(range(layer-2, layer+1)), param_ids=[6], layer_id=layer,
        module_str='_blocks({model_name})[{layer_id}]' if gemma else '{model_name}.model.layers[{layer_id}]',
        lr=5e-5, alpha=[1200.,1200.], steering_coeff_list=[6.5,6.5],
        max_num_batches=150, batch_size=4, seed=42, min_len=0, max_len=2000, verbose=False)
    params = namespace['get_params'](updated, settings.layer_ids, settings.param_ids)
    selected = {id(p) for p in params}
    names = [n for n,p in updated.named_parameters() if id(p) in selected]
    if len(params) != 3 or len(names) != 3 or any('mlp.down_proj' not in n for n in names):
        raise ValueError(f'Unexpected selected parameters: {names}')
    initial = [p.detach().cpu().clone() for p in params]
    report = dict(method='rmu-author', base_model=MODELS[args.variant], model=str(base),
        concept='joint', concepts=['wmdp-bio','wmdp-cyber'], git_commit=_git_commit(),
        author_revision=REVISION, source_hashes={k:v[1] for k,v in FILES.items()},
        config=vars(settings), data=data, selected_parameters=names, dtype=str(updated.dtype),
        optimizer='Transformers 4.38.2 AdamW, original class/defaults',
        package_versions={p:version(p) for p in ('torch','transformers','datasets')},
        source_fingerprint=before,
        adaptations=['local frozen corpus paths', 'one model copy per GPU and input placement',
                     'modern model runtime with original legacy optimizer', 'local metrics and separate evaluation'] +
                    (['native Gemma decoder paths', 'depth-scaled layer', 'nested hidden size',
                      'named down projections'] if gemma else []))
    (work/'request.json').write_text(json.dumps(report, indent=2))

    namespace['run_rmu'](updated, frozen, tokenizer, forget, retain, settings)
    if metrics.steps != 150: raise ValueError(f'Expected 150 updates; observed {metrics.steps}')
    if not all(bool(torch.isfinite(p).all()) for p in params):
        raise ValueError('Non-finite updated parameters; final registration refused')
    report['changed_elements'] = {n:int((p.detach().cpu()!=old).sum())
                                 for n,p,old in zip(names,params,initial)}
    report['completed_steps'] = metrics.steps
    if _fingerprint(base) != before: raise RuntimeError('Base model files changed')
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work/'checkpoint'), str(dst))
    report['copied_serving_files'] = complete_checkpoint(base, dst)
    report['carried_tensors'] = carry_over_missing(base, dst)
    (dst/'unlearn_report.json').write_text(json.dumps(report, indent=2))
    print(f'RMU author checkpoint saved: {dst}; changed elements={report["changed_elements"]}')
    if not all(report['changed_elements'].values()):
        print('WARNING: at least one selected tensor did not change; inspect before claiming successful training')


if __name__ == '__main__': main()
