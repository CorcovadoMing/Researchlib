from collections import namedtuple
import torch


class PSNR(namedtuple('PSNR', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach().view_as(x)
        # Assume the images is with range [-1, 1], rescale to [0, 1]
        x = (x + 1) / 2
        y = (y + 1) / 2
        return 10 * (1 / (x - y).pow(2).mean()).log10()
