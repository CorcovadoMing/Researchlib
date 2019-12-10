import numpy as np
import random
from .preprocessing import preprocessing


def _process_single(
    dp,
    normalizer,
    data_key = 0,
    label_key = 1,
    injector = None,
    transpose = ('NHWC', 'NCHW')
):
    x = np.array(dp[data_key]).astype(np.float32).copy()
    y = np.array(dp[label_key])
    
    if injector is not None:
        x = injector(x)

    # Deal with y type
    y_type = str(y.dtype)
    if 'int' in y_type:
        y_type = np.int64
    else:
        y_type = np.float32
    y = np.array(y).astype(y_type)

    # Normalize
    for op in normalizer:
        x = op(x)

    # Transfer to right format for pytorch
    if transpose is not None:
        x = preprocessing.format_transpose(x, *transpose)
        
    return x, y
