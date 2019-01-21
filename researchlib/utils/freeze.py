def module_trainable(module, requires_grad):
    for i, j in module.named_children():
        try:
            j.weight.requires_grad = requires_grad
        except:
            module_trainable(j, requires_grad)
