import numpy as np
import random


def _process_single(dp, is_train, include_y, normalizer, augmentor, data_key=0, label_key=1):
    x = dp[data_key].copy()
    y = dp[label_key]
    
    if include_y:
        y = y.copy()

    y_type = str(y.dtype)
    if 'int' in y_type:
        y_type = np.int64
    else:
        y_type = np.float32

    for op in normalizer:
        x = op.augment(x)

    if is_train:
        shuffled_augmentations = augmentor[:]
        random.shuffle(shuffled_augmentations)
        for i in range(len(x)):
            for op in shuffled_augmentations:
                x[i] = op.augment(x[i])

    return np.moveaxis(x, -1, 1).astype(np.float32), np.array(y).astype(y_type)