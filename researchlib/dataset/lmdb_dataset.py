from tensorpack.dataflow import *
from .general_dataset import _GeneralLoader
import cv2


def _LMDBDataset(file_path, shuffle = True, name = '', shuffle_queue = 50000, total_shuffle = False):
    df = LMDBSerializer.load(file_path, shuffle = total_shuffle)
    df = MapDataComponent(df, lambda x: cv2.imdecode(x, cv2.IMREAD_UNCHANGED), 0)
    if shuffle:
        df = LocallyShuffleData(df, shuffle_queue)
    return _GeneralLoader(df, name)
    