class _Captured(Exception):
    pass


def block_output(model, block, batch):

    captured = []

    def capture(_module, _inputs, output):
        captured.append(output[0] if isinstance(output, tuple) else output)
        raise _Captured

    handle = block.register_forward_hook(capture)
    try:
        from src.unlearn.precision import forward_precision
        with forward_precision(model):
            model(**batch, use_cache=False)
    except _Captured:
        return captured[0]
    finally:
        handle.remove()
    raise RuntimeError("selected decoder block was not executed")
