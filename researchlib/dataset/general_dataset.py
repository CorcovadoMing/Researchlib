import numpy as np
from tensorpack.dataflow import *
import cv2
from functools import partial


class _GeneralLoader:
    def __init__(self, ds, name):
        self.ds = ds
        self.name = name
        
    def get_generator(self, **kwargs):
        return self.ds

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
