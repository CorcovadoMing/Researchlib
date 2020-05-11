from tensorpack.dataflow import *
from .general_dataset import _GeneralLoader
import cv2


def _cv2_decode(data):
    return [cv2.imdecode(data[0], cv2.IMREAD_COLOR)] + list(data[1:])

def _LMDBDataset(file_path, shuffle = True, name = '', shuffle_queue = 10000, total_shuffle = False, workers = 2):
    df = LMDBSerializer.load(file_path, shuffle = total_shuffle)
    if workers >= 0:
        df = MultiProcessMapDataZMQ(df, workers, _cv2_decode)
    else:
        df = MapData(df, _cv2_decode)
    if shuffle:
        df = LocallyShuffleData(df, shuffle_queue)
    return _GeneralLoader(df, name)
    