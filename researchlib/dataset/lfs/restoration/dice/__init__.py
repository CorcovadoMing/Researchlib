from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def DiceMix(shuffle = True, resize = None):
    x = np.load(os.path.join(this_dir, 'mix_x.npy'))
    y = np.load(os.path.join(this_dir, 'mix_y.npy'))
    return _NumpyDataset(x, y, shuffle = shuffle, name = '', size = resize)