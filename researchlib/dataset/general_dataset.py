import numpy as np
from tensorpack.dataflow import *
import cv2
from functools import partial


def _transform(dp, size, bgr2rgb, data_key = 0, label_key = 1):
    x = np.array(dp[data_key]).astype(np.float32).copy()
    y = np.array(dp[label_key])
    
    if y.shape[:-1] == x.shape[:-1]:
        do_y = True
    else:
        do_y = False
        
    if size is not None:
        x = cv2.resize(x, (size, size))
        if do_y:
            y = cv2.resize(y, (size, size))
            
    if x.ndim == 2:
        x = x[..., None]
        if y.ndim == 2 and do_y:
            y = y[..., None]
        
    if bgr2rgb:
        x = x[:, :, (2,1,0)]
    return x, y


class _GeneralLoader:
    def __init__(self, ds, name, size = None, bgr2rgb = False):
        self.ds = ds
        self.name = name
        self.size = size
        self.bgr2rgb = bgr2rgb

    def get_generator(self, **kwargs):
        ds = self.ds
        _transform_fn = partial(_transform, size = self.size, bgr2rgb = self.bgr2rgb)
        ds = MapData(ds, _transform_fn)
        return ds

    def get_support_set(self, classes = [], shot = 5):
        collect = {k: [] for k in classes}
        g = self.get_generator(epochs = 3)
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
