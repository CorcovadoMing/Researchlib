def get_aux_out(model):
    result = []
    for i, j in model.named_children():
        try:
            result.append(j.store)
            j.store = []
        except:
            pass
    return result