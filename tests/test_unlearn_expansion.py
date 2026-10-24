import shlex
from pathlib import Path

import pytest

from scripts.plan_wmdp8_unlearn import PROFILES, commands, document
from src.paths import UNLEARN_BASES
from src.unlearn.r2mu_data import R2MU_MODELS, trace_text


def test_olmo_native_and_prefilled_trace_boundaries():
    common=dict(model='Olmo-3-7B-Think',index=0)
    assert 'reason' in trace_text(dict(common,raw_output='<think>reason</think>answer'))
    row=dict(common,prompt='user\nassistant\n<think>\n',raw_output='reason</think>answer')
    before=dict(row)
    assert 'reason' in trace_text(row)
    assert row==before
    with pytest.raises(ValueError,match='channel'):
        trace_text(dict(common,prompt='user mentions <think> elsewhere',raw_output='answer'))


def test_roster_excludes_completed_and_qwen_runs():
    runs={f'{method}-{code}' for code,method,_,_ in PROFILES}
    assert runs=={'r2mu-e4b','r2mu-g12','r2mu-g31','r2mu-o7','r2mu-o32',
                  'rmu-o7','rmu-o32','elm-o7','elm-o32',
                  'elm-e4b','elm-g12','elm-o7t','elm-o32t'}
    assert all('Qwen' not in UNLEARN_BASES[r] for r in runs)
    for run in runs:
        assert Path(f'configs/{run}.yaml').is_file()
        if run.startswith('r2mu'):assert UNLEARN_BASES[run] in R2MU_MODELS


def test_paired_jobs_keep_same_inputs_and_resources_and_smokes_separate():
    train=commands('train');smoke=commands('smoke')
    assert len(train)==36 and len(smoke)==26 and len(commands('traces'))==10
    parsed={}
    for line in train+smoke:
        parts=shlex.split(line)
        env=dict(x.split('=',1) for x in parts[parts.index('-v')+1].split(','))
        assert env['NAME'] not in parsed
        parsed[env['NAME']]=(parts,env)
        if '32' in env['MODEL'] or '31B' in env['MODEL']:
            assert 'gpus=2,gpu_memory=140G' in parts
        if env['CONCEPT']=='wmdp-bio' and 'DATA' not in env:
            assert env['FORGET_FILE']=='data/raw/bio-forget-corpus.jsonl'
    for name,(parts,env) in parsed.items():
        if not name.endswith('-t01'):continue
        other_parts,other=parsed[name.removesuffix('-t01')+'-control']
        assert other_parts[other_parts.index('-l')+1]==parts[parts.index('-l')+1]
        omit={'NAME','TRACE_WEIGHT','COT_WEIGHT'}
        assert {k:v for k,v in env.items() if k not in omit}=={k:v for k,v in other.items() if k not in omit}
        assert other['TRACE_WEIGHT']==other['COT_WEIGHT']=='0'


def test_checked_in_runbook_matches_command_source():
    assert Path('UNLEARN_EXPANSION.md').read_text()==document()


def test_submit_holds_only_r2mu_on_matching_trace_job():
    lines=commands('submit')
    assert len(lines)==46
    trace_names={shlex.split(line)[shlex.split(line).index('-N')+1] for line in lines[:10]}
    for line in lines[10:]:
        p=shlex.split(line)
        if p[-1]=='scripts/qsub_r2mu.sh':
            assert p[p.index('-hold_jid')+1] in trace_names
        else:
            assert '-hold_jid' not in p
    assert all('smoke' not in line for line in lines)
