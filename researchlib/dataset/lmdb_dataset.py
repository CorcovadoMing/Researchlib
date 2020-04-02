from tensorpack.dataflow import *
from .general_dataset import _GeneralLoader


def _LMDBDataset(file_path, shuffle = True, name = ''):
    df = LMDBSerializer.load(file_path, shuffle = shuffle)
    return _GeneralLoader(df, name)
    