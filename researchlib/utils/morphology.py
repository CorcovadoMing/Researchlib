import torch.nn.functional as F


def binary_erosion(inp, filter_size=3):
    return 1 - F.max_pool2d(
        1 - inp,
        kernel_size=filter_size,
        stride=1,
        padding=int((filter_size - 1) / 2))


def binary_dilation(inp, filter_size=3):
    return F.max_pool2d(
        inp,
        kernel_size=filter_size,
        stride=1,
        padding=int((filter_size - 1) / 2))
