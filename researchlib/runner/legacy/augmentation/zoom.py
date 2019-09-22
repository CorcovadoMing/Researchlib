from ..template import template
from ...utils import mapping
import numpy as np
import cv2


class Zoom(template.NumpyAugmentation):
    def __init__(self, prob = None, mag = None, include_y = False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag

    def _aug_fn(self, img, zoom_factor):
        new_img = np.zeros_like(img)
        img = img.transpose(1, 2, 0)
        img_size = (int(img.shape[0] * zoom_factor), int(img.shape[1] * zoom_factor))
        img = cv2.resize(img, img_size)
        img = img.transpose(2, 0, 1)

        if zoom_factor > 1:
            padding_h = (img.shape[1] - new_img.shape[1]) // 2
            padding_w = (img.shape[2] - new_img.shape[2]) // 2
            new_img = img[:, padding_h:padding_h + new_img.shape[1], padding_w:padding_w +
                          new_img.shape[2]]
        else:
            padding_h = (new_img.shape[1] - img.shape[1]) // 2
            padding_w = (new_img.shape[2] - img.shape[2]) // 2
            new_img[:, padding_h:padding_h + img.shape[1], padding_w:padding_w +
                    img.shape[2]] = img

        return new_img

    def forward_single(self, x, y, mag):
        zoom_factor = mapping(mag, [0, 1], [0.5, 2], to_int = False)
        x = [self._aug_fn(i, zoom_factor) for i in x]
        if self.include_y:
            y = [self._aug_fn(i, zoom_factor) for i in y]
        return x, y
