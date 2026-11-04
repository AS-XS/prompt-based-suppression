#!/usr/bin/env python3


import argparse
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import UNLEARN_BASES

H48 = 'h_rt=48:00:00'
MAX_SECONDS = 169200
CARD = {'small': 'gpus=1,gpu_memory=80G', 'mid': 'gpus=1,gpu_memory=140G',
        'large': 'gpus=2,gpu_memory=80G'}
BIO_FILE = 'data/raw/bio-forget-corpus.jsonl'
TRACES = 'data/processed/r2mu-training'
DOMAINS = ('bio', 'cyber')
METHODS = ('rmu', 'r2mu', 'elm')


ROSTER = {
    'rmu': (('e4b', 'rmu-e4b', 'small'), ('g12', 'rmu-g12', 'mid'), ('o7', 'rmu-o7', 'small'),
            ('o32', 'rmu-o32-mp80', 'large'), ('g31', 'rmu-g31-mp80', 'large')),
    'r2mu': (('e4b', 'r2mu-e4b', 'small'), ('g12', 'r2mu-g12', 'mid'), ('o7', 'r2mu-o7', 'small'),
             ('g31', 'r2mu-g31-mp80', 'large'), ('o32', 'r2mu-o32-mp80', 'large')),
    'elm': (('e4b', 'elm-e4b', 'small'), ('g12', 'elm-g12', 'mid')),
}
NEW_NAMESPACES = ('rmu-g31-mp80',)
ELM_WINDOWS = (('w48', 'rel:4-8'), ('w812', 'rel:8-12'), ('w1216', 'rel:12-16'))


def paper_samples(domain, card):


    if card == 'large':
        return 500
    return 2000 if domain == 'bio' else 960


def paper_data(code, domain, card):
    return f'{TRACES}/{code}-{domain}-n{paper_samples(domain, card)}'


def trace_job_name(code, domain):
    return f'r2mu-{code}-{domain}-paper-traces'


def checkpoint(method, code, domain, suffix, card):
    return f'{method}-{code}-{domain}-{suffix}' + ('-mp80' if card == 'large' else '')


def qsub(name, env, script, resources, hold=None):
    parts = ['qsub'] + (['-hold_jid', hold.replace('-', '_')] if hold else [])
    parts += ['-l', resources, '-N', name.replace('-', '_'),
              '-v', ','.join(f'{k}={v}' for k, v in env.items()), script]
    return shlex.join(parts)


def _forget(env, domain):
    if domain == 'bio':
        env['FORGET_FILE'] = BIO_FILE
    return env


def rmu_jobs():
    for code, run, card in ROSTER['rmu']:
        large = card == 'large'
        for domain in DOMAINS:
            name = checkpoint('rmu', code, domain, 'p02', card)
            env = _forget(dict(MODEL=UNLEARN_BASES[run], NAME=name, CONCEPT=f'wmdp-{domain}'), domain)

            env.update(LAYER='auto', STEERING_RATIO=1.1, ALPHA=100, LR=0.00025,
                       MAX_BATCHES=600 if large else 150, BATCH_SIZE=1 if large else 4,
                       FROZEN_DTYPE='bfloat16', EARLY_STOP=1)
            if large:
                env.update(DTYPE='bfloat16', MIXED_PRECISION=1, FROZEN_DEVICE='cuda:1')
            else:
                env['DTYPE'] = 'float32'
            yield run, name, qsub(name, env, 'scripts/qsub_unlearn.sh',
                                  CARD[card] + (',' + H48 if large else ''))


def trace_jobs():
    for code, run, card in ROSTER['r2mu']:
        for domain in DOMAINS:
            env = _forget(dict(MODE='traces', MODEL=UNLEARN_BASES[run], CONCEPT=f'wmdp-{domain}',
                               DATA=paper_data(code, domain, card),
                               SAMPLES=paper_samples(domain, card)), domain)
            name = trace_job_name(code, domain)
            yield run, name, qsub(name, env, 'scripts/qsub_r2mu.sh', CARD['small'] + ',' + H48)


def r2mu_jobs():
    for code, run, card in ROSTER['r2mu']:
        large = card == 'large'
        for domain in DOMAINS:
            common = dict(MODEL=UNLEARN_BASES[run], CONCEPT=f'wmdp-{domain}', LAYER='auto',
                          STEERING_RATIO=1.1, BATCH_SIZE=1 if large else 4)
            if large:
                common.update(FROZEN_DEVICE='cuda:1', MIXED_PRECISION=1)
            rungs = (

                ('g100', dict(DATA=f'{TRACES}/{code}-{domain}', GAMMA=100, LR=0.00025, STEPS=150),
                 ('t01', 0.1, 1), None),
                ('paper', dict(DATA=paper_data(code, domain, card), GAMMA=1.4, LR=0.000075,
                               STEPS=500, FIT_STEPS=1),
                 ('full', 1, 1), trace_job_name(code, domain)),
            )
            for rung, settings, (label, trace, cot), hold in rungs:

                for suffix, trace_weight, cot_weight in ((label, trace, cot), ('control', 0, 0)):
                    name = checkpoint('r2mu', code, domain, f'{rung}-{suffix}', card)
                    env = dict(common, NAME=name, **settings, TRACE_WEIGHT=trace_weight,
                               COT_WEIGHT=cot_weight, RESUME=1, MAX_SECONDS=MAX_SECONDS)
                    yield run, name, qsub(name, env, 'scripts/qsub_r2mu.sh',
                                          CARD[card] + ',' + H48, hold)


def elm_jobs():
    for code, run, card in ROSTER['elm']:
        for domain in DOMAINS:
            for label, window in ELM_WINDOWS:
                name = checkpoint('elm', code, domain, f'e1000-{label}', card)
                env = _forget(dict(MODEL=UNLEARN_BASES[run], NAME=name, CONCEPT=f'wmdp-{domain}'),
                              domain)
                env.update(LAYERS=window, ETA=1000, SAMPLES=3000, LR=0.00005, RESUME=1,
                           MAX_SECONDS=MAX_SECONDS)
                yield run, name, qsub(name, env, 'scripts/qsub_elm.sh', CARD[card] + ',' + H48)


TRAINING = {'rmu': rmu_jobs, 'r2mu': r2mu_jobs, 'elm': elm_jobs}


def checkpoints(methods=METHODS):

    out = {}
    for method in methods:
        for run, name, _ in TRAINING[method]():
            out.setdefault(run, []).append(name)
    return out


def commands(stage, methods=METHODS):
    if stage == 'setup':
        return ['python scripts/setup_unlearn_runs.py ' + ' '.join(NEW_NAMESPACES)]
    if stage == 'traces':
        return [line for _, _, line in trace_jobs()] if 'r2mu' in methods else []
    if stage == 'train':
        return [line for method in methods for _, _, line in TRAINING[method]()]
    if stage == 'submit':
        return commands('traces', methods) + commands('train', methods)
    groups = checkpoints(methods)
    if stage == 'generate':
        out = []
        for run, names in groups.items():
            model = UNLEARN_BASES[run]

            shards = ' SHARDS=8' if '32B' in model and model.endswith('Think') else ''
            out.append(f'MODELS_DIR=models/unlearned GPU_MEM=80 GPU_UTIL=0.92{shards} '
                       f'WAVES="none-baseline" CONFIG=configs/{run}.yaml bash scripts/generate_all.sh '
                       + ' '.join(names))
        return out
    if stage == 'score':

        out = ['for run in ' + ' '.join(groups) + '; do [ ! -d results/$run/analysis ] || '
               '[ -d results/$run/analysis-round1 ] || '
               'cp -R results/$run/analysis results/$run/analysis-round1; done']
        for run, names in groups.items():
            model = UNLEARN_BASES[run]
            out.append(f'python scripts/coverage.py --run {run} --config configs/{run}.yaml '
                       f'--models {model} ' + ' '.join(names))
            out.append(f'python scripts/score_accuracy.py --run {run} --split val --baseline-model {model}')
        return out
    raise ValueError(stage)


PREAMBLE = '''# Unlearning round 2 — 2026-11-04

Generated by `python scripts/plan_unlearn_round2.py --markdown`. Nothing is submitted locally.

**Why.** Round 1 gave every new model the 31B pilot's settings (steering ratio 1.1, retain weight
300, lr 5e-5). That is weaker than the one Gemma setting that removed anything (E2B RMU: retain
100, lr 2.5e-4). For R2MU it is ~200x the authors' retain weight (1.4) with 30% of their updates.
The papers tune every model; none promises removal at a transferred setting.

| family | names | settings |
|---|---|---|
| RMU | `-p02` | E2B's working setting: ratio 1.1, alpha 100, lr 2.5e-4; 150 updates x 4 (31B/32B: 600 x 1) |
| R2MU | `-g100-t01`, `-g100-control` | E2B's R2MU setting: gamma 100, lr 2.5e-4, trace 0.1, CoT 1, 150 updates |
| R2MU | `-paper-full`, `-paper-control` | the authors' released scale: gamma 1.4, lr 7.5e-5, trace 1, CoT 1, up to 500 updates |
| ELM | `-e1000-w48`, `-w812`, `-w1216` | eta 1000, rank 4, 3000 samples; layers 4-8, 8-12, 12-16 of 32, scaled to depth |

Models: RMU on Gemma E4B / 12B / 31B and OLMo 7B / 32B Instruct; R2MU on Gemma E4B / 12B / 31B
and OLMo 7B / 32B Think; ELM on Gemma E4B / 12B (the paper's config is unchanged for OLMo).
31B/32B use the mixed-precision 2x80G path at batch 1 and write to `-mp80` namespaces; 31B RMU
gets a new one, `rmu-g31-mp80`, because rmu4 holds bf16 checkpoints.

**What is kept.** Every round-1 checkpoint, trace bundle, generation and namespace is untouched:
the names are new, the trainers refuse a completed checkpoint, and the paper rung prepares its
traces in new directories. Cyber's forget corpus has only 1,000 documents, so its paper rung
prepares 960 traces and `--fit-steps` caps updates at what the usable traces fill, as the
authors' trainer does. The steering coefficient stays 1.1 x RMS everywhere.

**Order.** Run setup once after `git pull`. Submit queues the R2MU paper-rung trace jobs first;
the jobs that use those traces hold on them. Everything else starts immediately. Generate only
after a checkpoint is exported; score after coverage is complete; select on validation under
§8e's gate. Leave test untouched.
'''

LABELS = {'setup': 'Set up the one new namespace (after git pull)',
          'submit': 'Submit (trace preparation first; dependent jobs hold on it)',
          'generate': 'Generate after the checkpoints are exported',
          'score': 'Coverage and validation scoring after generation'}


def document():
    text = PREAMBLE
    for stage, label in LABELS.items():
        text += f'\n## {label}\n'
        if stage in ('submit', 'generate', 'score'):
            for method in METHODS:
                text += f'\n### {method.upper()}\n\n```bash\n' + '\n\n'.join(commands(stage, (method,))) + '\n```\n'
        else:
            text += '\n```bash\n' + '\n\n'.join(commands(stage)) + '\n```\n'
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=['setup', 'traces', 'train', 'submit', 'generate', 'score'])
    ap.add_argument('--method', choices=METHODS, action='append',
                    help='Method.')
    ap.add_argument('--markdown', action='store_true')
    args = ap.parse_args()
    if not args.markdown and not args.stage:
        ap.error('choose --stage or --markdown')
    if args.markdown:
        print(document(), end='')
    else:
        print('\n'.join(commands(args.stage, tuple(args.method or METHODS))))
