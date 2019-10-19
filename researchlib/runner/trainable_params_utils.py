from collections import defaultdict


def group_by_key(items):
    res = defaultdict(list)
    for k, v in items:
        res[k].append(v)
    return res


trainable_params = lambda model: {k: p for k, p in model.named_parameters() if p.requires_grad}
is_bias = lambda model: group_by_key(('bias' in k, v) for k, v in trainable_params(model).items())
num_model_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
num_list_params = lambda model: sum(p.numel() for p in model if p.requires_grad)
