def set_lr(opt, lr):
    def _inner_process(opt, lr):
        for g in opt.param_groups:
            g['lr'] = lr

    if type(opt) == type([]):
        for i in opt:
            _inner_process(i, lr)
    else:
        _inner_process(opt, lr)
