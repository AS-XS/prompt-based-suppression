import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.unlearn.r2mu_author import author_tree, verified_source, load_author


def test_limo_repair_is_row_local_and_loop_is_unchanged():
    source = '''
def get_limo_dataset(dataset):
    for ex in dataset:
        user_msg = ex['messages'][0]['content']
        assistant_list.append(assistant_msg)
def run_rmu():
    loss = unlearn_loss + retain_loss + extent_loss + args.assist_loss * assistant_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''
    tree = author_tree(source)
    loop = tree.body[0].body[0]
    scope = {'assistant_list': []}
    exec(compile(tree, 'test-author', 'exec'), scope)
    scope['get_limo_dataset']([
        {'messages': [{'role':'user','content':'q1'}, {'role':'assistant','content':'a1'}]},
        {'messages': [{'role':'user','content':'q2'}, {'role':'assistant','content':'a2'}]}])
    assert scope['assistant_list'] == ['a1','a2']
    assert ast.dump(tree.body[1]) == ast.dump(ast.parse(source).body[1])


def test_unverified_source_is_refused(tmp_path):
    (tmp_path/'utils.py').write_text('raise RuntimeError("must never execute")')
    with pytest.raises(ValueError, match='hash mismatch'):
        verified_source(tmp_path, 'utils.py')


def test_gemma_changes_only_architecture_and_delimiters():
    source = '''
def get_limo_dataset(dataset):
    for ex in dataset:
        user_msg = 0
        assistant_list.append(assistant_msg)
def run_rmu():
    width = updated_model.config.hidden_size
    prefix = "Let's reason this step by step.\\n<think>"
    suffix = "{}</think>\\n\\n"
    loss.backward()
'''
    code = ast.unparse(author_tree(source, gemma=True))
    assert '_hidden_size(updated_model.config)' in code
    assert '<think>' not in code and '</think>' not in code
    assert '<|channel>thought' in code and '<channel|>' in code
    assert 'loss.backward()' in code


@pytest.fixture
def upstream():
    path = os.environ.get('R2MU_AUTHOR_TEST_SOURCE')
    if not path:
        pytest.skip('Set R2MU_AUTHOR_TEST_SOURCE to verified downloaded sources for integration test')
    return Path(path)


def test_real_upstream_loop_is_preserved_and_updates_weights(upstream, tmp_path, monkeypatch):
    import copy
    import sys
    torch = pytest.importorskip('torch')

    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=None,AutoModelForCausalLM=None))
    logs = []
    logger = SimpleNamespace(init=lambda **kw: None, log=logs.append)
    ns = load_author(upstream, logger)
    raw = ast.parse(verified_source(upstream,'unlearn_wmdp.py'))
    transformed = author_tree(verified_source(upstream,'unlearn_wmdp.py'))
    loop = lambda tree: next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_rmu')
    assert ast.dump(loop(raw)) == ast.dump(loop(transformed))

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
        def save_pretrained(self,path): Path(path).mkdir(parents=True,exist_ok=True)
    class Batch(dict):
        def to(self, device): return self
    class Tokenizer:
        def __call__(self, texts, **kw): return Batch(input_ids=torch.tensor([[0,1]]*len(texts)))
        def save_pretrained(self,path): pass
    model = Model()
    frozen = copy.deepcopy(model)
    before = [p.clone() for p in model.parameters()]
    args = SimpleNamespace(output_dir=str(tmp_path),wandb_group='test',layer_ids=[5,6,7],
        param_ids=[0],layer_id=7,module_str='{model_name}.model.layers[{layer_id}]',
        lr=7.5e-5,steering_coeff_list=[6.5],alpha=[1.4],assist_loss=1,max_num_batches=2)
    data = [[['a','b'],['c','d']]]
    result = ns['run_rmu'](model,frozen,Tokenizer(),data,data,data,data,data,data,args)
    assert Path(result).is_dir() and len(logs)==2
    changed = [not torch.equal(a,b) for a,b in zip(before,model.parameters())]
    assert changed == [False]*5+[True]*3
    assert all(torch.equal(a,b) for a,b in zip(before,frozen.parameters()))


def test_real_limo_loader_uses_each_assistant_solution(upstream, monkeypatch):
    import sys
    pytest.importorskip('torch')
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=None,AutoModelForCausalLM=None))
    class Dataset(list):
        def map(self, fn, batched):
            rows = fn({'question':['q1','q2'],'solution':['a1','a2']})
            return Dataset([{'messages':m} for m in rows['messages']])
        def remove_columns(self, names): return self
    monkeypatch.setitem(sys.modules,'datasets',SimpleNamespace(load_dataset=lambda *a,**k:Dataset()))
    ns = load_author(upstream, SimpleNamespace())
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages,**kw: messages[0]['content'])
    _, assistant, _ = ns['get_limo_dataset'](tokenizer,batch_size=2)
    assert assistant[0][0] == ['a1</think>\n\n','a2</think>\n\n']
    assert len(assistant[0]) == 3
