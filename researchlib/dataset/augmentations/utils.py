import numpy as np
import PIL


def _to_pil(x):
    return PIL.Image.fromarray(x.astype(np.uint8))


def _to_numpy(x):
    return np.array(x)