import numpy as np
from tensorpack.dataflow import *
import cv2
from functools import partial


def _resize(dp, size, data_key = 0, label_key = 1):
    x = np.array(dp[data_key]).astype(np.float32).copy()
    y = np.array(dp[label_key])
    return cv2.resize(x, (size, size)), y


class _GeneralLoader:
    def __init__(self, ds, name, resize = None):
        self.ds = ds
        self.name = name
        self.resize = resize

    def get_generator(self, batch_size = 512, **kwargs):
        ds = self.ds
        if 'fixed_batch' in kwargs:
            ds = FixedSizeData(ds, batch_size * kwargs['fixed_batch'], keep_state=False)
            ds = LocallyShuffleData(ds, batch_size * kwargs['fixed_batch'])
        if self.resize is not None:
            resize_fn = partial(_resize, size = self.resize)
            ds = MapData(ds, resize_fn)
        ds = BatchData(ds, batch_size, remainder = True)
        return ds

    def get_support_set(self, classes = [], shot = 5):
        collect = {k: [] for k in classes}
        g = self.get_generator(1, epochs = 3)
        g.reset_state()
        count = 0
        for x, y in g:
            x = x[0]
            y = y[0]
            if y in classes and len(collect[y]) < shot:
                collect[y].append(x)
                count += 1
            if count == len(classes) * shot:
                break
        del g
        return np.stack(list(collect.values())), np.repeat(np.array(list(collect.keys())),
                                                           shot).reshape(len(classes), shot)
