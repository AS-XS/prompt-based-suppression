import ast
import hashlib
from pathlib import Path

REVISION = 'c0b6c12bb0de3decf9c13bb13f9f1aa15754a132'
ROOT = Path('data/external/rmu-author')
FILES = {
    'unlearn.py': (f'https://raw.githubusercontent.com/centerforaisafety/wmdp/{REVISION}/rmu/unlearn.py',
                   'd27dec8915eed0c7b1fb0ad7498f02ed33a353e168cd3229dd14f010ecf917c2'),
    'utils.py': (f'https://raw.githubusercontent.com/centerforaisafety/wmdp/{REVISION}/rmu/utils.py',
                 'bf26d906957873b1f07d70099fcfb9b6b8d4680a8f7955646c52bbff314d193c'),
    'zephyr.ipynb': (f'https://raw.githubusercontent.com/centerforaisafety/wmdp/{REVISION}/run_rmu_zephyr.ipynb',
                    '35195e80c5527fef9f66f85767414e7f9a5d31e05f4abdb45871012e3436d148'),
    'optimization.py': ('https://raw.githubusercontent.com/huggingface/transformers/v4.38.2/src/transformers/optimization.py',
                        '65b8816c5919360049b7a6cf348c3052c0c2a566f5921c80a8e70756c26bc2ff'),
}
MODELS = {'zephyr': 'zephyr-7b-beta', 'g12': 'gemma-4-12B-it',
          'llama': 'Meta-Llama-3-8B-Instruct'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(root, name):
    path = Path(root) / name
    if sha(path) != FILES[name][1]:
        raise ValueError(f'Author source hash mismatch: {path}')
    return path.read_text()


def function_tree(text, names, gemma=False):
    tree = ast.parse(text)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in names]
    if {n.name for n in tree.body} != set(names):
        raise ValueError(f'Missing upstream declarations: {names}')
    if gemma:
        class HiddenSize(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if ast.unparse(node) == 'updated_model.config.hidden_size':
                    return ast.copy_location(ast.parse('_hidden_size(updated_model.config)',
                                                       mode='eval').body, node)
                return self.generic_visit(node)
        tree = HiddenSize().visit(tree)
    return ast.fix_missing_locations(tree)


def load_author(root=ROOT, gemma=False):
    import datetime
    import math
    import warnings
    from typing import Callable, Iterable, Tuple
    import torch
    import tqdm
    from transformers.utils.versions import require_version
    from src.unlearn.rmu import _decoder_layers, hidden_size_of
    ns = dict(__name__='rmu_author', torch=torch, nn=torch.nn, Optimizer=torch.optim.Optimizer,
              math=math, warnings=warnings, Callable=Callable, Iterable=Iterable, Tuple=Tuple,
              require_version=require_version, datetime=datetime, tqdm=tqdm,
              _blocks=_decoder_layers, _hidden_size=hidden_size_of)
    for name, functions in [('optimization.py', ['AdamW']),
                            ('utils.py', ['forward_with_cache', 'get_params']),
                            ('unlearn.py', ['run_rmu'])]:
        exec(compile(function_tree(source(root, name), functions, gemma), name, 'exec'), ns)
    original_forward = ns['forward_with_cache']
    def placed_forward(model, inputs, module, no_grad=True):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        return original_forward(model, inputs, module, no_grad=no_grad)
    ns['forward_with_cache'] = placed_forward
    if gemma:
        ns['get_params'] = lambda model, layers, ids: [
            _decoder_layers(model)[i].mlp.down_proj.weight for i in layers]
    return ns
