import json
from pathlib import Path

import pytest

from src.unlearn.r2mu_data import digest, load_bundle, resume_rows, trace_text


def test_raw_preserved_and_explicit_channel_required():
    row = dict(index=0, raw_output='<|channel>analysis\nreasoning<channel|>answer')
    before = dict(row)
    assert 'reasoning' in trace_text(row)
    assert row == before
    with pytest.raises(ValueError, match='channel'):
        trace_text(dict(index=1, raw_output='I cannot answer.'))


def test_preparation_integrity_and_missing_trace_filter(tmp_path):
    manifest = dict(model='gemma-4-E2B-it', concept='wmdp-bio', samples=2, config_hash='abc')
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    rows = [dict(index=i, config_hash='abc', forget='training text',
                 source_hash=digest('training text'), finish_reason='stop', raw_output=raw)
            for i,raw in enumerate(['<|channel>analysis\ntrace<channel|>answer','refusal'])]
    path=tmp_path/'traces.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    _, selected, traces = load_bundle(tmp_path, 'gemma-4-E2B-it','wmdp-bio')
    assert [r['index'] for r in selected] == [0] and len(traces)==1
    with pytest.raises(ValueError, match='model/concept'):
        load_bundle(tmp_path,'gemma-4-31B-it','wmdp-bio')
    path.write_text(json.dumps(rows[0])+'\n')
    with pytest.raises(ValueError, match='Incomplete'):
        load_bundle(tmp_path,'gemma-4-E2B-it','wmdp-bio')
    rows[0]['forget']='changed'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='hash mismatch'):
        load_bundle(tmp_path,'gemma-4-E2B-it','wmdp-bio')


def test_interrupted_trace_write_recovery(tmp_path):
    path=tmp_path/'raw.jsonl';path.write_bytes(b'{"index":0}\n{"index":')
    assert resume_rows(path)==[{'index':0}]
    assert path.read_bytes()==b'{"index":0}\n'
    path.write_bytes(b'{"index":0}')
    assert resume_rows(path)==[{'index':0}]
    assert path.read_bytes().endswith(b'\n')
    path.write_bytes(b'bad\n{"index":0}\n')
    with pytest.raises(ValueError, match='Corrupt'):
        resume_rows(path)


def test_masked_loss_ignores_padding_and_sequential_gradients_match_joint():
    torch=pytest.importorskip('torch')
    from src.unlearn.r2mu import masked_mse
    p=torch.tensor([[[1.,2.],[100.,100.]]],requires_grad=True)
    mask=torch.tensor([[1,0]])
    loss=masked_mse(p,torch.zeros_like(p),mask)
    assert loss.item()==2.5
    loss.backward();assert torch.equal(p.grad[0,1],torch.zeros(2))
    parameter=torch.nn.Parameter(torch.tensor([1.,2.]))
    batches=[torch.tensor([float(i),float(i+1)]) for i in range(4)]
    weights=[1.,100.,1.,1.]
    joint=sum(w*((parameter*x).square().mean()) for x,w in zip(batches,weights))
    joint.backward();expected=parameter.grad.clone();parameter.grad=None
    for x,w in zip(batches,weights):
        (w*(parameter*x).square().mean()).backward()
    assert torch.allclose(parameter.grad,expected)


def test_new_eval_configs_match_existing_instrument():
    import yaml
    base=yaml.safe_load(Path('configs/rmu2.yaml').read_text())
    for run in ('r2mu-e2b','r2mu-g31'):
        config=yaml.safe_load(Path(f'configs/{run}.yaml').read_text())
        assert config.pop('run')==run
        assert config=={k:v for k,v in base.items() if k!='run'}


def test_training_engine_runs_all_four_losses_and_exports(tmp_path, monkeypatch):
    torch=pytest.importorskip('torch')
    import sys
    from types import SimpleNamespace, ModuleType
    from src.unlearn import r2mu, corpora
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp=torch.nn.Module(); self.mlp.down_proj=torch.nn.Linear(3,3,bias=False)
        def forward(self,x):
            return self.mlp.down_proj(x)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model=torch.nn.Module()
            self.model.layers=torch.nn.ModuleList([Layer(),Layer(),Layer()])
            self.config=SimpleNamespace(hidden_size=3)
        def forward(self,input_ids,attention_mask,use_cache=False):
            x=input_ids.float().unsqueeze(-1).expand(-1,-1,3)
            for block in self.model.layers: x=block(x)
            return x
        def save_pretrained(self,dst,**kw):
            (Path(dst)/'weights.txt').write_text('saved')
    initial=Model().state_dict()
    class Loader:
        @staticmethod
        def from_pretrained(*a,**kw):
            m=Model();m.load_state_dict(initial);return m
    class Batch(dict):
        def to(self,device): return self
    class Tokenizer:
        pad_token_id=0
        @classmethod
        def from_pretrained(cls,*a,**kw): return cls()
        def __call__(self,texts,**kw):
            return Batch(input_ids=torch.tensor([[len(t)%7+1,2] for t in texts]),
                         attention_mask=torch.ones(len(texts),2))
        def save_pretrained(self,dst): pass
    module=ModuleType('transformers')
    module.AutoModelForCausalLM=Loader;module.AutoTokenizer=Tokenizer
    monkeypatch.setitem(sys.modules,'transformers',module)
    monkeypatch.chdir(tmp_path)
    src=tmp_path/'models'/'gemma-4-E2B-it';src.mkdir(parents=True)
    (src/'config.json').write_text('{}')
    root=tmp_path/'data';root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(model=src.name,concept='wmdp-bio',
        samples=2,config_hash='abc',base_fingerprint=r2mu._fingerprint(src))))
    rows=[dict(index=i,forget='corpus',source_hash=digest('corpus'),config_hash='abc',
        raw_output='<|channel>analysis\nreasoning<channel|>end',finish_reason='stop') for i in range(2)]
    (root/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    limo=root/'limo.jsonl';limo.write_text(json.dumps({'solution':'math reasoning'})+'\n')
    monkeypatch.setattr(corpora,'load_retain',lambda **kw:['retain','other retain'])
    monkeypatch.setattr(r2mu,'complete_checkpoint',lambda *a:None)
    monkeypatch.setattr(r2mu,'carry_over_missing',lambda *a:None)
    args=SimpleNamespace(models_dir='models',model=src.name,name='r2mu-test',concept='wmdp-bio',
        data=root,limo_file=limo,steps=2,batch_size=1,layer=2,coeff=6.,gamma=1.,trace_weight=1.,
        cot_weight=1.,lr=.001,seed=42,frozen_dtype='float32',frozen_device='cpu',device='cpu',
        max_seconds=0,resume=False)
    r2mu.run(args)
    report=json.loads(Path('models/unlearned/r2mu-test/unlearn_report.json').read_text())
    assert report['method']=='r2mu' and report['base_model']==src.name
    assert set(report['losses'])=={'forget','retain','trace','cot','total'}
    assert all(len(v)==2 for v in report['losses'].values())
    assert report['losses']['trace'][0]>0
    assert Path('results/r2mu-training/r2mu-test/resume.pt').is_file()
    with pytest.raises(ValueError,match='complete'):
        r2mu.run(args)
