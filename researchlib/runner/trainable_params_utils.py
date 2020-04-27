trainable_params = lambda model: {k: p for k, p in model.named_parameters() if p.requires_grad}
num_model_params = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
num_list_params = lambda model: sum(p.numel() for p in model if p.requires_grad)


def group_parameters(model, bias_key=['bias'], no_decay_key=['coefficients'], special_key=[]):
    normal_group = []
    bias_group = []
    no_decay_group = []
    special_group = []
    for k, v in trainable_params(model).items():
        if sum([k.find(i) > -1 for i in bias_key]) > 0:
            bias_group.append(v)
        elif sum([k.find(i) > -1 for i in no_decay_key]) > 0:
            no_decay_group.append(v)
        elif sum([k.find(i) > -1 for i in special_key]) > 0:
            special_group.append(v)
        else:
            normal_group.append(v)
    return normal_group, bias_group, no_decay_group, special_group
        
    