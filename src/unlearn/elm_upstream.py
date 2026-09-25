import ast
import hashlib

REVISION = "fa3306415f7b3cd0eb37c8a2838f11bb361c3143"
SOURCE_SHA256 = "b5af7f76d8f8d8d63b389d656e2e3d48d4afc48f320bcc0face8de86b259dd35"


def adapted_tree(source: str):
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("ELM source hash mismatch; do not run an unreviewed revision")
    tree = ast.parse(source)
    kept = []
    for node in tree.body:
        if isinstance(node, ast.If):
            continue
        if isinstance(node, ast.ImportFrom) and node.module in {
            "utils.lora", "utils.metrics", "lm_eval", "lm_eval.models.huggingface",
        }:
            continue
        if isinstance(node, ast.Import) and any(
            x.name in {"lm_eval", "wandb", "matplotlib.pyplot"} for x in node.names
        ):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "train_elm":
            body = []
            loaded = moved = saved = False
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
                        and ast.unparse(stmt.targets[0]) == "model"):
                    fn = ast.unparse(stmt.value.func)
                    if fn == "AutoModelForCausalLM.from_pretrained":
                        stmt.value = ast.parse("_load_model(model_id, dtype)", mode="eval").body
                        loaded = True
                    elif fn == "model.to":
                        moved = True
                        continue
                body.append(stmt)
                if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                        and ast.unparse(stmt.value.func) == "model.save_pretrained"):
                    body += ast.parse("return model, tokenizer, filename, losses").body
                    saved = True
                    break
            if not (loaded and moved and saved):
                raise ValueError("ELM adapter no longer matches the reviewed training structure")
            node.body = body
        kept.append(node)
    tree.body = kept
    class AdapterTargets(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == "get_peft_model":
                node.func.id = "_get_elm_peft_model"
                node.args.append(ast.Name(id="get_peft_model", ctx=ast.Load()))
            return node

    return ast.fix_missing_locations(AdapterTargets().visit(tree))
