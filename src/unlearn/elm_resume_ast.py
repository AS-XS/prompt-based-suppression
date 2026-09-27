import ast


def add_recovery(function):
    matches = 0
    body = []
    for stmt in function.body:


        if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == 'os.path.exists(filename)':
            continue
        if isinstance(stmt, ast.With):
            loops = [n for n in stmt.body if isinstance(n, ast.For)
                     and ast.unparse(n.target) == 'idx'
                     and ast.unparse(n.iter) == 'range(0, batchsize)']
            if loops:
                if len(loops) != 1:
                    raise ValueError('Unexpected ELM loop structure')
                loop = loops[0]
                body += ast.parse(
                    '_resume_start, iter_cnt, dataset_cntr, losses = '
                    '_recovery.restore(model, optimizer)').body
                loop.iter = ast.parse('range(_resume_start, batchsize)', mode='eval').body


                loop.body = [n for n in loop.body if not (
                    isinstance(n, ast.If) and ast.unparse(n.test) == 'args.save_every is not None')]
                loop.body += ast.parse(
                    '_recovery.boundary(model, optimizer, idx + 1, iter_cnt, '
                    'dataset_cntr, losses, accumulation_steps, batchsize)').body
                stmt.body.insert(0, ast.parse('pbar.update(_resume_start)').body[0])
                matches += 1
        body.append(stmt)
    if matches != 1:
        raise ValueError('ELM recovery no longer matches the pinned training loop')
    function.body = body
