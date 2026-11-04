import shlex
from pathlib import Path

import pytest
import yaml

from scripts import plan_unlearn_80g, plan_wmdp8_unlearn
from scripts.plan_unlearn_round2 import (ROSTER, checkpoints, commands, document, paper_samples,
                                          trace_job_name)
from src.paths import UNLEARN_BASES
from src.unlearn.elm import layer_window
from src.unlearn.r2mu import fitted_steps


def parse(line):
    parts = shlex.split(line)
    env = dict(x.split('=', 1) for x in parts[parts.index('-v') + 1].split(','))
    hold = parts[parts.index('-hold_jid') + 1] if '-hold_jid' in parts else None
    return parts, env, hold


def names_of(lines):
    return [parse(line)[1]['NAME'] for line in lines]


def test_checked_in_runbook_matches_command_source():
    assert Path('UNLEARN_ROUND2.md').read_text() == document()


def test_round_one_names_and_namespaces_are_never_reused():
    new = names_of(commands('train'))
    assert len(new) == len(set(new)) == 10 + 40 + 12
    old = set(names_of(plan_wmdp8_unlearn.commands('train'))) | set(names_of(plan_unlearn_80g.commands('train')))
    assert not set(new) & old
    for run in checkpoints():
        assert run in UNLEARN_BASES and Path(f'configs/{run}.yaml').is_file()


def test_large_models_use_the_mp80_path_and_namespaces():
    for method, roster in ROSTER.items():
        for code, run, card in roster:
            assert (card == 'large') == run.endswith('-mp80') == (code in ('g31', 'o32'))
    for line in commands('train'):
        parts, env, _ = parse(line)
        if env['NAME'].endswith('-mp80'):
            assert parts[parts.index('-l') + 1].startswith('gpus=2,gpu_memory=80G')
            assert env['MIXED_PRECISION'] == '1' and env['FROZEN_DEVICE'] == 'cuda:1'
            assert env.get('BATCH_SIZE', '1') == '1'


def test_rmu_is_e2bs_working_setting_everywhere():
    for line in commands('train', ('rmu',)):
        _, env, hold = parse(line)
        assert (env['ALPHA'], env['LR'], env['STEERING_RATIO']) == ('100', '0.00025', '1.1')
        assert hold is None and env['NAME'].split('-')[3] == 'p02'
        assert int(env['MAX_BATCHES']) * int(env['BATCH_SIZE']) == 600
        assert env['DTYPE'] == ('bfloat16' if 'MIXED_PRECISION' in env else 'float32')
        if env['CONCEPT'] == 'wmdp-bio':
            assert env['FORGET_FILE'] == 'data/raw/bio-forget-corpus.jsonl'


def test_r2mu_rungs_pair_with_matched_controls_and_wait_for_their_traces():
    traces = {parse(line)[0][parse(line)[0].index('-N') + 1]: parse(line)[1]
              for line in commands('traces')}
    assert len(traces) == 10
    jobs = {env['NAME']: (parts, env, hold) for parts, env, hold in map(parse, commands('train', ('r2mu',)))}
    assert len(jobs) == 40
    for name, (parts, env, hold) in jobs.items():
        rung = 'paper' if '-paper-' in name else 'g100'
        if rung == 'paper':
            prep = traces[hold]
            assert env['DATA'] == prep['DATA'] and env['FIT_STEPS'] == '1'
            assert (env['GAMMA'], env['LR'], env['STEPS']) == ('1.4', '7.5e-05', '500')
            assert prep['MODEL'] == env['MODEL'] and prep['CONCEPT'] == env['CONCEPT']
        else:
            assert hold is None and 'FIT_STEPS' not in env
            assert (env['GAMMA'], env['LR'], env['STEPS']) == ('100', '0.00025', '150')
            assert not env['DATA'].split('/')[-1].split('-')[-1].startswith('n')
        if '-control' not in name:
            continue
        full = name.replace('-control', '-full' if rung == 'paper' else '-t01')
        other_parts, other, other_hold = jobs[full]
        omit = {'NAME', 'TRACE_WEIGHT', 'COT_WEIGHT'}
        assert {k: v for k, v in env.items() if k not in omit} == {k: v for k, v in other.items() if k not in omit}
        assert other_parts[other_parts.index('-l') + 1] == parts[parts.index('-l') + 1] and other_hold == hold
        assert env['TRACE_WEIGHT'] == env['COT_WEIGHT'] == '0'
        assert other['TRACE_WEIGHT'] == ('1' if rung == 'paper' else '0.1')


def test_paper_traces_fit_the_corpora_and_never_reuse_round_one_bundles():
    assert paper_samples('cyber', 'small') <= 1000  
    for line in commands('traces'):
        _, env, _ = parse(line)
        assert env['DATA'].endswith(f"-n{env['SAMPLES']}")
        assert ('FORGET_FILE' in env) == (env['CONCEPT'] == 'wmdp-bio')
    submit = commands('submit')
    assert submit[:10] == commands('traces')
    assert {trace_job_name(c, d).replace('-', '_') for c, _, _ in ROSTER['r2mu'] for d in ('bio', 'cyber')} \
        == {shlex.split(line)[shlex.split(line).index('-N') + 1] for line in submit[:10]}


def test_elm_sweeps_three_windows_at_eta_1000():
    jobs = [parse(line)[1] for line in commands('train', ('elm',))]
    assert len(jobs) == 12
    assert {j['LAYERS'] for j in jobs} == {'rel:4-8', 'rel:8-12', 'rel:12-16'}
    assert all(j['ETA'] == '1000' and j['SAMPLES'] == '3000' for j in jobs)


def test_generation_and_scoring_cover_every_new_checkpoint_once():
    names = names_of(commands('train'))
    generate, score = commands('generate'), commands('score')
    for name in names:
        assert sum(f' {name}' in line.split('generate_all.sh')[1] for line in generate) == 1
        assert sum(line.startswith('python scripts/coverage.py') and f' {name}' in line for line in score) == 1
    assert all(('SHARDS=8' in line) == ('r2mu-o32-mp80.yaml' in line) for line in generate)
    assert score[0].startswith('for run in ') and 'analysis-round1' in score[0]


def test_new_31b_namespace_matches_rmu4s_instrument():
    base = yaml.safe_load(Path('configs/rmu4.yaml').read_text())
    config = yaml.safe_load(Path('configs/rmu-g31-mp80.yaml').read_text())
    assert config.pop('run') == 'rmu-g31-mp80' and base.pop('run') == 'rmu4'
    assert config == base and UNLEARN_BASES['rmu-g31-mp80'] == 'gemma-4-31B-it'


@pytest.mark.parametrize('depth', range(26, 65))
def test_layer_window_auto_is_round_ones_formula(depth):
    assert layer_window('auto', depth) == (round(depth * 4 / 32), round(depth * 8 / 32))
    assert layer_window('4,8', depth) == (4, 8)
    start, end = layer_window('rel:12-16', depth)
    assert 0 <= start < end <= depth


def test_layer_window_rejects_empty_or_out_of_range():
    with pytest.raises(ValueError):
        layer_window('rel:8-8', 32)
    with pytest.raises(ValueError):
        layer_window('30,40', 32)


def test_fitted_steps_caps_at_the_data_like_the_authors():
    assert fitted_steps(500, 1800, 4) == 450
    assert fitted_steps(150, 800, 4) == 150
    with pytest.raises(ValueError):
        fitted_steps(500, 3, 4)
