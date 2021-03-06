def _merge_dict(x, y):
    result = {}
    for i in x:
        result[i] = x[i]
    for i in y:
        if i in result:
            result[i] += y[i]
        else:
            result[i] = y[i]
    return result


def get_reg_out(model_):
    result = {}
    for i, j in model_.named_children():
        try:
            if j.reg_group not in result:
                result[j.reg_group] = []
            result[j.reg_group].append(j.reg_store)
            j.reg_store = []
        except:
            pass
        r_ = get_reg_out(j)
        result = _merge_dict(result, r_)
    return result
