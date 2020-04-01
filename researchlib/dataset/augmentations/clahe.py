import numpy as np
import cv2


class CLAHE:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        
    def __call__(self, x, choice):
        if choice:
            if x.shape[-1] != 1:
                x = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2LAB)
                x[..., 0] = self.clahe.apply(x[..., 0])
                return cv2.cvtColor(x, cv2.COLOR_LAB2RGB).astype(np.float32)
            else:
                x = x.astype(np.uint8)
                x[..., 0] = self.clahe.apply(x[..., 0])
                return x.astype(np.float32)
        else:
            return x
        
    def options(self, prob=0.5):
        return [{
            'choice': b,
        } for b in np.random.choice([True, False], p=[prob, 1-prob], size=1)]



