def set_lr(opt, lr, key = 'lr'):
    def _inner_process(opt, lr, key):
        for g in opt.param_groups:
            g[key] = lr

    if type(opt) == list or type(opt) == tuple:
        if type(lr) == list or type(lr) == tuple:
            for i, j in zip(opt, lr):
                _inner_process(i, j, key)
        else:
            for i in opt:
                _inner_process(i, lr, key)
    else:
        _inner_process(opt, lr, key)
