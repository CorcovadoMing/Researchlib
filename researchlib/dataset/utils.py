import numpy as np


def _PaddingSequence(_y, value, max_length):
    y = np.zeros((_y.shape[0], max_length))
    y += value
    y[:, :_y.shape[1]] = _y
    return y.astype(np.int)


class utils(object):
    PaddingSequence = _PaddingSequence