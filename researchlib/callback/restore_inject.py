def restore_inject(model, node):
    target_node = model[node]
    model[node] = (target_node[0].orig_f, *target_node[1:])