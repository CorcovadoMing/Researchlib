def get_aux_out(model):
    result = []
    def _inner(m):
        try:
            result.append(m.store)
            m.store = None
        except:
            pass

    model.apply(_inner)
    return result
