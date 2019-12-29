from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def Noise2d(name, is_train, shuffle = True, resize = None):
    total = ['erlang', 'exponential', 'gaussian', 'lognormal', 'poisson', 'rayleigh', 'saltpepper', 'speckle', 'uniform']
    if name not in total:
        raise ValueError(f'{name} is not in {total}')
    phase = 'train' if is_train else 'test'
    x = np.load(os.path.join(this_dir, f'{name}_{phase}.npy'), allow_pickle=True)
    return _NumpyDataset(x, x.copy(), shuffle = shuffle, name = '', size = resize)
