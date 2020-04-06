import numpy as np
from collections import namedtuple


def _svd_blur(img, K):
    rec = []
    for k in range(img.shape[-1]):
        data = img[..., k]
        u, s, v = np.linalg.svd(data, full_matrices=True)
        s[-K:] = 0
        rec.append(((u * s) @ v))
    rec = np.stack(rec, -1)
    return rec


class SVDBlur(namedtuple('SVDBlur', ('img_size'))):
    def __call__(self, x, choice, K):
        return _svd_blur(x, K) if choice else x

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'K': np.random.choice(range(self.img_size - (self.img_size//8)))
        }



