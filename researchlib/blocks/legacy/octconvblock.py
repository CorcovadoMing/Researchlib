from torch import nn
from ..layers import layer


class _OctConvBlock2d(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride = 1,
        padding = 0,
        alphas = (0.75, 0.75),
        pool = False,
        pooling_factor = 2,
        activator = nn.ReLU,
        norm = nn.BatchNorm2d
    ):
        super().__init__()
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_in <= 1, "Alphas must be in interval [0, 1]"

        # CH IN
        self.ch_in_hf = int((1 - self.alpha_in) * ch_in)
        self.ch_in_lf = ch_in - self.ch_in_hf

        # CH OUT
        self.ch_out_hf = int((1 - self.alpha_out) * ch_out)
        self.ch_out_lf = ch_out - self.ch_out_hf

        self.conv = layer.OctConv2d(ch_in, ch_out, kernel_size, stride, padding, alphas)
        self.norm_hf = norm(self.ch_out_hf)
        if self.ch_out_lf:
            self.norm_lf = norm(self.ch_out_lf)
        self.activator = activator()
        self.pool = pool
        if self.pool:
            self.pooling = nn.MaxPool2d(pooling_factor)

    def forward(self, x):
        if self.ch_out_lf > 0:
            h, l = self.conv(x)
            h, l = self.activator(self.norm_hf(h)), self.activator(self.norm_lf(l))
            if self.pool:
                h, l = self.pooling(h), self.pooling(l)
            return h, l
        else:
            h = self.conv(x)
            h = self.activator(self.norm_hf(h))
            if self.pool:
                h = self.pooling(h)
            return h
