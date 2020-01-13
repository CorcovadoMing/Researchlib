from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def COIL100(name, is_train, shuffle = True, resize = None):
    if is_train:
        x = np.load(os.path.join(this_dir, f'{name}_x.npy'))
        y = np.load(os.path.join(this_dir, f'{name}_y.npy'))
    else:
        x = np.load(os.path.join(this_dir, f'{name}_testx.npy'))
        y = np.load(os.path.join(this_dir, f'{name}_testy.npy'))
    return _NumpyDataset(x, y, shuffle = shuffle, name = '', size = resize)
