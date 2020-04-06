from tensorpack.dataflow import *
from .general_dataset import _GeneralLoader


def _LMDBDataset(file_path, shuffle = True, name = '', shuffle_queue = 50000):
    df = LMDBSerializer.load(file_path, shuffle = False)
    if shuffle:
        df = LocallyShuffleData(df, shuffle_queue)
    return _GeneralLoader(df, name)
    