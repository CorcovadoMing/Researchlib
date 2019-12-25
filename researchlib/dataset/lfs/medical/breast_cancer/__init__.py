from ....np_dataset import _NumpyDataset
import numpy as np
import os


this_dir, _ = os.path.split(__file__)


def BreastCancer(shuffle = True, resize = None):
    x = np.load(os.path.join(this_dir, 'train_img.npy'))[:, 0, :, :]
    y = np.load(os.path.join(this_dir, 'train_label.npy'))
    return _NumpyDataset(x, y, shuffle = shuffle, name = '', size = resize)