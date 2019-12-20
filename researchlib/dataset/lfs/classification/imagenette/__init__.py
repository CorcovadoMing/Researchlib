import numpy as np
import os
from tensorpack.dataflow import *
from ....general_dataset import _GeneralLoader


this_dir, _ = os.path.split(__file__)


def ImageNetteFull(is_train, shuffle = True, resize = None):
    file = 'train.lmdb' if is_train else 'val.lmdb'
    file_path = os.path.join(this_dir, 'fullsize', file)
    df = LMDBSerializer.load(file_path, shuffle = shuffle)
    return _GeneralLoader(df, 'imagenette_fullsize', resize, True)
    