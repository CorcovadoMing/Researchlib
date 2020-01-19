from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def BSD68(name, is_train, shuffle = True, merge_train_val = True, resize = None):
    x = np.load(os.path.join(this_dir, f'{name}_x.npy'), allow_pickle=True)
    y = np.load(os.path.join(this_dir, f'{name}_y.npy'), allow_pickle=True)
    
    return _NumpyDataset(x, y, shuffle = shuffle, name = '', size = resize)
