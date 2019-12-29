from collections import namedtuple
import torch


class PSNR(namedtuple('PSNR', [])):
    def __call__(self, x, y):
        x, y = x.detach(), y.detach().view_as(x)
        return 10 * (1 / (x - y).pow(2).mean()).log10()
