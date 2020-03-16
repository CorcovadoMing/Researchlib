from torch import nn


class _AEDecoder2d(nn.Module):
    def __init__(self, blocks, dimension, to_rgb=True, to_grayscale=False):
        super().__init__()
        model = []
        cur_dim = dimension
        for _ in range(blocks):
            model += [
                nn.ConvTranspose2d(cur_dim, cur_dim//2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(cur_dim//2),
                nn.ReLU()
            ]
            cur_dim = cur_dim//2
        assert not (to_rgb and to_grayscale)
        if to_rgb:
            model.append(nn.Conv2d(cur_dim, 3, 1, bias=False))
        if to_grayscale:
            model.append(nn.Conv2d(cur_dim, 1, 1, bias=False))
        self.f = nn.Sequential(*model)
    
    def forward(self, x):
        return self.f(x)