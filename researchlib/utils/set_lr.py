def set_lr(opt, lr):
    def _inner_process(opt, lr):
        for g in opt.param_groups:
            g['lr'] = lr

    if type(opt) == list or type(opt) == tuple:
        if type(lr) == list or type(lr) == tuple:
            for i, j in zip(opt, lr):
                _inner_process(i, j)
        else:
            for i in opt:
                _inner_process(i, lr)
    else:
        _inner_process(opt, lr)
