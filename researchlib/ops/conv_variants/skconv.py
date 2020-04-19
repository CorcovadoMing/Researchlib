from torch import nn
import torch
import torch.nn.functional as F


class _SKConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, M=2, G=32, r=16, L=32, **kwargs):
        super().__init__()
        from ...blocks.unit.conv import _Conv
        
        self.convs = nn.ModuleList([])
        for i in range(M):
            conv_kwargs = {
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding+i,
                'dilation': dilation+i,
                'groups': G,
                'bias': bias
            }
            self.convs.append(_Conv('__sk', nn.Conv2d, in_channels, out_channels, **conv_kwargs))
        
        d = max(int(out_channels/r), L)
        self.extractor = nn.Sequential(
            nn.Linear(out_channels, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(inplace=True)
        )
        
        self.transform = nn.ModuleList([])
        for i in range(M):
            self.transform.append(nn.Linear(d, out_channels, bias=False))
        
    def forward(self, x):
        features = torch.stack([i(x) for i in self.convs], dim=1)
        U = features.sum(1)
        s = U.mean(-1).mean(-1)
        z = self.extractor(s)
        att = torch.stack([i(z) for i in self.transform], dim=1)
        att = F.softmax(att, 1).view(*att.size(), 1, 1)
        return (features * att).sum(1)