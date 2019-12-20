from ...np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)

def Meow32(shuffle = True):
    x = np.load(os.path.join(this_dir, 'meow32.npy'))
    return _NumpyDataset(x, x.copy(), shuffle = shuffle, name = '')
                         
def Meow64(shuffle = True):
    x = np.load(os.path.join(this_dir, 'meow64.npy'))
    return _NumpyDataset(x, x.copy(), shuffle = shuffle, name = '')

def Meow128(shuffle = True):
    x = np.load(os.path.join(this_dir, 'meow128.npy'))
    return _NumpyDataset(x, x.copy(), shuffle = shuffle, name = '')