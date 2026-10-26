import shlex
from pathlib import Path

import pytest

from scripts.plan_unlearn_80g import commands, RUNS
from src.paths import UNLEARN_BASES


def test_mp80_commands_are_independent_and_reuse_only_trace_inputs():
    lines=commands('train')
    assert len(lines)==14
    names=[]
    for line in lines:
        p=shlex.split(line)
        env=dict(x.split('=',1) for x in p[p.index('-v')+1].split(','))
        assert p[p.index('-l')+1]=='gpus=2,gpu_memory=80G,h_rt=48:00:00'
        assert env['NAME'].endswith('-mp80')
        assert env['MIXED_PRECISION']=='1'
        if p[-1]!='scripts/qsub_unlearn.sh':assert env['MAX_SECONDS']=='169200'
        if 'DATA' in env:assert not env['DATA'].endswith('-mp80')
        names.append(env['NAME'])
    assert len(set(names))==14
    generated='\n'.join(commands('generate'))
    scored='\n'.join(commands('score'))
    for n in names:assert n in generated and n in scored
    for run in RUNS:
        assert UNLEARN_BASES[run+'-mp80']==UNLEARN_BASES[run]
        assert Path(f'configs/{run}-mp80.yaml').is_file()
        assert f'--run {run}-mp80 ' in scored


def test_tiny_updates_keep_float32_master_weights_and_resume(tmp_path):
    torch=pytest.importorskip('torch')
    from src.unlearn.precision import promote_trainable, forward_precision
    from src.unlearn.elm_resume import Recovery
    model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.Linear(2,1,bias=False)).bfloat16()
    model[0].requires_grad_(False)
    model[1].weight.data.fill_(1)
    frozen=model[0].weight.detach().clone()
    promote_trainable(model)
    assert model[0].weight.dtype==torch.bfloat16
    assert model[1].weight.dtype==torch.float32
    opt=torch.optim.AdamW([model[1].weight],lr=0.00005)
    before=model[1].weight.detach().clone()
    with forward_precision(model):
        loss=model(torch.ones(1,2,dtype=torch.bfloat16)).float().square().sum()
    loss.backward(); opt.step(); opt.zero_grad()
    assert not torch.equal(before,model[1].weight)
    assert torch.equal(before.bfloat16(),model[1].weight.bfloat16())
    assert torch.equal(frozen,model[0].weight)
    assert opt.state[model[1].weight]['exp_avg'].dtype==torch.float32
    rec=Recovery(tmp_path,{'precision':'mixed'},resume=True,seconds=0)
    rec.boundary(model,opt,1,0,{}, {},1,1)
    expected=model[1].weight.detach().clone()
    model[1].weight.data.zero_()
    rec.restore(model,opt)
    assert torch.equal(model[1].weight,expected)
    with pytest.raises(RuntimeError,match='mismatch'):
        Recovery(tmp_path,{'precision':'float32'},resume=True).restore(model,opt)
