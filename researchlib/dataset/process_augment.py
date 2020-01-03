import numpy as np
import random
from .preprocessing import preprocessing


def _process_augment(
    dp,
    include_y,
    augmentor,
    data_key = 0,
    label_key = 1,
):
    x = dp[data_key].copy()
    y = dp[label_key]

    # Augmentation
    for op in augmentor:
        options = op.options()
        x = op(x, **random.choice(options))

    return x, y
