from contextlib import nullcontext


def promote_trainable(model):
    import torch
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.float32)
            count += parameter.numel()
    if not count:
        raise ValueError('mixed precision requires selected trainable parameters')
    model._unlearn_mixed_precision = True
    print(f'Mixed precision: bf16 backbone/compute; {count} float32 trainable parameters', flush=True)


def forward_precision(model):
    if not getattr(model, '_unlearn_mixed_precision', False):
        return nullcontext()
    import torch
    device = next(model.parameters()).device.type
    return torch.autocast(device_type=device, dtype=torch.bfloat16)
