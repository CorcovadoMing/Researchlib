from functools import partial
from .normalize import normalize
from .denormalize import denormalize
from .dynamic_normalize import DynamicNormalize


def set_normalizer(type, mean, std):
    if type == 'static':
        normalizer = partial(normalize, mean = mean, std = std)
        denormalizer = partial(denormalize, mean = mean, std = std)
    else:
        normalizer = DynamicNormalize()
        denormalizer = None # TODO

    #return normalizer, denormalizer
    return normalizer
