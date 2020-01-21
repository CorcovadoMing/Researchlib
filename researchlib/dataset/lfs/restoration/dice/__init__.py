from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def Dice(name, shuffle = True, resize = None):
    if name not in ['mix', '1000', '950', '900', '850', '800', '750', '1000nac', '1000nactest']:
        raise
    x = np.load(os.path.join(this_dir, f'res{name}_x.npy'))
    y = np.load(os.path.join(this_dir, f'res{name}_y.npy'))
    return _NumpyDataset(x, y, shuffle = shuffle, name = '', size = resize)
