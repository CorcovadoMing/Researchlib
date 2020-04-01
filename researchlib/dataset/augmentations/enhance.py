import numpy as np
from skimage.filters import unsharp_mask


class Enhance:
    def __init__(self):
        pass
        
    def __call__(self, x, choice):
        return unsharp_mask(x) if choice else x
        
    def options(self, prob=0.5):
        return [{
            'choice': b,
        } for b in np.random.choice([True, False], p=[prob, 1-prob], size=1)]



