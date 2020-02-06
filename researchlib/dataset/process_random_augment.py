import numpy as np
import random
from .preprocessing import preprocessing


def _process_random_augment(
    dp,
    N, 
    M,
    include_y,
    augmentor,
    data_key = 0,
    label_key = 1,
):
    x = dp[data_key].copy()
    y = dp[label_key]

    # Augmentation
    for op in random.choices(augmentor, k=N):
        options = op.options(prob=1)
        x = op(x, **random.choice(options))

    return x, y
