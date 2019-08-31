def _switch_swa_mode(optimizer):
    try:
        if type(optimizer) == list:
            for i in optimizer:
                i.swap_swa_sgd()
        else:
            optimizer.swap_swa_sgd()
    except:
        pass