class _Config:
    pass

def build_config(**kwargs):
    _new_config = _Config()
    for i in kwargs:
        setattr(_new_config, i, kwargs[i])
    return _new_config