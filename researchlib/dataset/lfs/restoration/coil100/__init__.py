from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def COIL100(name, is_train, shuffle = True, merge_train_val = True, resize = None):
    if is_train:
        if merge_train_val:
            x_test = np.load(os.path.join(this_dir, f'{name}_testx.npy')) 
            y_test = np.load(os.path.join(this_dir, f'{name}_testy.npy'))
            if 'nac' in name:
                x_test + (0.3 * np.random.normal(size=x_test.shape) * x_test)
            x = np.concatenate([np.load(os.path.join(this_dir, f'{name}_x.npy')), x_test], 0)
            y = np.concatenate([np.load(os.path.join(this_dir, f'{name}_y.npy')), y_test], 0)
        else:
            x = np.load(os.path.join(this_dir, f'{name}_x.npy'))
            y = np.load(os.path.join(this_dir, f'{name}_y.npy'))
    else:
        x = np.load(os.path.join(this_dir, f'{name}_testx.npy'))
        y = np.load(os.path.join(this_dir, f'{name}_testy.npy'))
    
    return _NumpyDataset(x, y, shuffle = shuffle, name = '', size = resize)
