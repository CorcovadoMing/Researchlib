from torch import nn


class _DilConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
        padding_mode = 'zeros'
    ):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels, 
                                    in_channels, 
                                    kernel_size, 
                                    stride=stride, 
                                    padding=2*padding, 
                                    dilation=2, 
                                    groups=in_channels, 
                                    bias=bias, 
                                    padding_mode=padding_mode)
        self.point_conv = nn.Conv2d(in_channels, 
                                    out_channels, 
                                    1, 
                                    bias=bias)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


