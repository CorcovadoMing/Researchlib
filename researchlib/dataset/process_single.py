import numpy as np
import random
from .preprocessing import preprocessing


def _process_single(
    dp,
    normalizer,
    data_key = 0,
    label_key = 1,
    injector = None,
    transpose = ('HWC', 'CHW')
):
    x = np.array(dp[data_key]).astype(np.float32).copy()
    y = np.array(dp[label_key]).astype(np.float32)
    
    if x.shape[:-1] == y.shape[:-1]:
        do_y = True
    else:
        do_y = False
    
    if injector is not None:
        x = injector(x)

    # Normalize
    for op in normalizer:
        x = op(x)
        if do_y:
            y = op(y)

    # Transfer to right format for pytorch
    if transpose is not None:
        x = preprocessing.format_transpose(x, *transpose)
        if do_y:
            y = preprocessing.format_transpose(y, *transpose)
            
    if do_y:
        y = np.array(y).astype(np.float32)
    else:
        y = np.array(y).astype(np.int64)
    
    return x, y
