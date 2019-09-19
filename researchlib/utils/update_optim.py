def update_optim(opt, value, key):
    def _inner_process(opt, value, key):
        for g in opt.param_groups:
            g[key] = value

    if type(opt) == list or type(opt) == tuple:
        if type(value) == list or type(value) == tuple:
            for i, j in zip(opt, value):
                _inner_process(i, j, key)
        else:
            for i in opt:
                _inner_process(i, value, key)
    else:
        _inner_process(opt, value, key)
