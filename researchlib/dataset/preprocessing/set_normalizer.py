from functools import partial
from .normalize import normalize
from .dynamic_normalize import DynamicNormalize


def set_normalizer(type, mean, std):
    if type == 'static':
        normalizer = [partial(normalize, mean = mean, std = std)]
    else:
        normalizer = [DynamicNormalize()]

    return normalizer
