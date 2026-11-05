import ast
import hashlib
from pathlib import Path

REVISION = '3529e557a9c5d72ef53b7ee2b677a3aa8d2c8b06'
HASHES = {
    'unlearn_wmdp.py': 'a6eca0da69435eda4534812714e4ae3fefbea9f2957ee539fcfeeb79c7d5939f',
    'utils.py': '214c693caf74a35527edf00031f3ec932f66eaa64d6f591a3b7486ed701e8119',
    'generated_all_wmdp.jsonl': '0d72755aa227d0324ae19ce0cd92f29f71af4f9faaa17ccc5991d5e49698c150',
}
DEFAULT_SOURCE = Path('data/external/r2mu-author')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_source(root, filename):
    path = Path(root) / filename
    if sha(path) != HASHES[filename]:
        raise ValueError(f'Upstream hash mismatch: {path}; rerun setup, do not edit pinned files')
    return path.read_text()


def author_tree(text, gemma=False):
    tree = ast.parse(text)

    tree.body = [n for n in tree.body if not isinstance(n, ast.If)
                 and not (isinstance(n, ast.ImportFrom) and n.module in {'utils', 'metrics'})
                 and not (isinstance(n, ast.Import) and any(a.name == 'wandb' for a in n.names))]
    limo = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_limo_dataset')
    loop = next(n for n in limo.body if isinstance(n, ast.For))

    loop.body.insert(1, ast.parse(
        'assistant_msg = [m["content"] for m in ex["messages"] if m["role"] == "assistant"][0]'
    ).body[0])
    if gemma:
        class Gemma(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if ast.unparse(node) == 'updated_model.config.hidden_size':
                    return ast.copy_location(ast.parse('_hidden_size(updated_model.config)', mode='eval').body, node)
                return self.generic_visit(node)
            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    value = node.value.replace("Let's reason this step by step.\n<think>", '<|channel>thought\n')
                    value = value.replace('</think>', '<channel|>')
                    return ast.copy_location(ast.Constant(value), node)
                return node
        tree = Gemma().visit(tree)
    return ast.fix_missing_locations(tree)


def load_author(root, logger, gemma=False):
    utils = {'__name__': 'r2mu_author_utils'}
    exec(compile(verified_source(root, 'utils.py'), str(Path(root)/'utils.py'), 'exec'), utils)
    namespace = {**utils, 'wandb': logger, '__name__': 'r2mu_author'}
    from src.unlearn.rmu import _decoder_layers, hidden_size_of
    namespace.update(_blocks=_decoder_layers, _hidden_size=hidden_size_of)
    text = verified_source(root, 'unlearn_wmdp.py')
    exec(compile(author_tree(text, gemma), str(Path(root)/'unlearn_wmdp.py'), 'exec'), namespace)
    return namespace
