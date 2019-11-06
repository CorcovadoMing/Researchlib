import numpy as np


def _ExtendSequence(_y, max_length, value = 0):
    pad = np.zeros((_y.shape[0], max_length - _y.shape[1], *_y.shape[2:])) + value
    print(_y.shape, pad.shape)
    return np.concatenate([_y, pad], axis=1)


class utils(object):
    ExtendSequence = _ExtendSequence