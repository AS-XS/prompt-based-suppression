import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.unlearn.rmu_author import function_tree, load_author, source
from src.unlearn.rmu_author_data import eligible, load_bundle


def test_source_hash_refuses_changes(tmp_path):
    (tmp_path/'unlearn.py').write_text('not the author source')
    with pytest.raises(ValueError, match='hash mismatch'): source(tmp_path, 'unlearn.py')


def test_gemma_patch_only_changes_nested_hidden_size():
    text = 'def run_rmu():\n x = updated_model.config.hidden_size\n loss.backward()\n optimizer.step()\n'
    adapted = function_tree(text, ['run_rmu'], gemma=True)
    assert ast.unparse(adapted) == ast.unparse(ast.parse(text.replace(
        'updated_model.config.hidden_size', '_hidden_size(updated_model.config)')))


def test_upstream_min_len_keeps_whitespace_and_long_documents():
    assert eligible(['', ' ', 'x'*2500]) == [' ', 'x'*2500]


def test_changed_frozen_bundle_is_refused(tmp_path):
    (tmp_path/'setup.json').write_text(json.dumps({'data_hashes': {'corpora.json': 'wrong', 'corpora-manifest.json': 'wrong'}}))
    (tmp_path/'corpora.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'): load_bundle(tmp_path)


@pytest.fixture
def upstream():
    root = os.environ.get('RMU_AUTHOR_TEST_SOURCE')
    if not root: pytest.skip('Set RMU_AUTHOR_TEST_SOURCE for pinned-source CPU integration')
    return Path(root)


def test_actual_author_loop_and_optimizer_update_only_selected_weights(upstream, tmp_path, monkeypatch):
    import copy
    torch = pytest.importorskip('torch')
    import sys
    monkeypatch.setitem(sys.modules, 'transformers.utils.versions',
                        SimpleNamespace(require_version=lambda requirement: None))
    ns = load_author(upstream)
    raw = ast.parse(source(upstream, 'unlearn.py'))
    loop = next(n for n in raw.body if isinstance(n, ast.FunctionDef) and n.name == 'run_rmu')
    assert ast.dump(loop) == ast.dump(function_tree(source(upstream, 'unlearn.py'), ['run_rmu']).body[0])
    original_class = next(n for n in ast.parse(source(upstream, 'optimization.py')).body
                          if isinstance(n, ast.ClassDef) and n.name == 'AdamW')
    assert ast.dump(original_class) == ast.dump(function_tree(
        source(upstream, 'optimization.py'), ['AdamW']).body[0])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(8)])
            self.config = SimpleNamespace(hidden_size=2)
        @property
        def dtype(self): return next(self.parameters()).dtype
        @property
        def device(self): return next(self.parameters()).device
        def forward(self, input_ids, **kw):
            x = torch.nn.functional.one_hot(input_ids,2).float()
            for layer in self.model.layers: x=layer(x)
            return x
        def save_pretrained(self, path): Path(path).mkdir(parents=True,exist_ok=True)
    calls = []
    class Batch(dict):
        def to(self, device): return self
    class Tokenizer:
        truncation_side = 'left'
        def __call__(self, texts, **kw):
            calls.append((texts[0], kw['max_length']))
            return Batch(input_ids=torch.tensor([[0,1]]*len(texts)))
        def save_pretrained(self, path): pass
    model = Model(); frozen = copy.deepcopy(model)
    initial = [p.detach().clone() for p in model.parameters()]
    optimizer = ns['AdamW']([next(model.parameters())], lr=5e-5)
    assert optimizer.defaults['eps'] == 1e-6 and optimizer.defaults['weight_decay'] == 0
    args = SimpleNamespace(output_dir=str(tmp_path/'out'), layer_ids=[5,6,7], param_ids=[0],
        layer_id=7,module_str='{model_name}.model.layers[{layer_id}]',lr=5e-5,
        steering_coeff_list=[6.5,6.5],alpha=[1200.,1200.],max_num_batches=2,verbose=False)
    bio = [['bio']*4]*2; cyber = [['cyber']*4]*2; retain = [['retain']*4]*2
    tok = Tokenizer()
    ns['run_rmu'](model,frozen,tok,[bio,cyber],[retain,retain],args)
    assert [not torch.equal(a,b) for a,b in zip(initial,model.parameters())] == [False]*5+[True]*3
    assert all(torch.equal(a,b) for a,b in zip(initial,frozen.parameters()))
    assert calls == [('bio',512),('retain',512),('cyber',768),('retain',512)]
    assert tok.truncation_side == 'left'


def test_setup_freezes_order_and_preserves_existing_bundle(tmp_path, monkeypatch):
    import sys
    from src.unlearn.rmu_author import sha
    from src.unlearn.rmu_author_data import write_bundle
    class Dataset:
        _fingerprint = 'fixture'
        def __getitem__(self, key): return [''] + [f'text {i}' for i in range(605)]
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **kw: Dataset()))
    bio = tmp_path/'bio.jsonl'
    bio.write_text(''.join(json.dumps({'text':f'bio {i}'})+'\n' for i in range(605)))
    write_bundle(tmp_path, bio)
    hashes = {name:sha(tmp_path/name) for name in ('corpora.json','corpora-manifest.json')}
    (tmp_path/'setup.json').write_text(json.dumps({'data_hashes':hashes}))
    forget, retain, _ = load_bundle(tmp_path)
    assert len(forget) == len(retain) == 2 and len(forget[0]) == 150
    assert forget[0][0] == ['bio 0','bio 1','bio 2','bio 3']
    assert retain[0][0] == ['text 0','text 1','text 2','text 3']
    write_bundle(tmp_path, bio)
    bio.write_text(bio.read_text().replace('bio 0','CHANGED'))
    with pytest.raises(ValueError, match='Differing frozen input'): write_bundle(tmp_path, bio)
    assert hashes['corpora.json'] == sha(tmp_path/'corpora.json')
