from PIL import Image
import numpy as np


class _PILHelper:

    def __init__(self):
        pass

    def to_pil(self, img):
        # Save min, max
        self.min = img.min()
        img -= img.min()
        self.max = img.max()
        img /= (img.max() + 1e-7)
        return Image.fromarray(np.uint8(img * 255))

    def to_numpy(self, img):
        img = np.array(img).astype(np.float32)
        img /= 255
        # Reverse to saved min, max
        img *= self.max
        img += self.min
        return img
