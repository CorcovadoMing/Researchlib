from torch import nn

class _RConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', step=3):
        super().__init__()
        self.ff_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.r_conv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.step = step
    
    def forward(self, x):
        ff_transform = self.ff_conv(x)
        out = ff_transform
        for _ in range(self.step):
            out = ff_transform + self.r_conv(out)
        return out     