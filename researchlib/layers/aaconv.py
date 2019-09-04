from torch import nn
import torch.nn.functional as F
import torch


class _MHA2d(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8):
        super().__init__()
        self.dk = out_dim
        self.dv = out_dim
        self.heads = heads
        self.wq = nn.Linear(in_dim, self.dk, bias=False)
        self.wk = nn.Linear(in_dim, self.dk, bias=False)
        self.wv = nn.Linear(in_dim, self.dv, bias=False)
        self.wo = nn.Linear(self.dv, self.dv, bias=False)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).transpose(-1, -2)
        wq = self.wq(x).view(b, h*w, -1, self.heads).permute((0,3,1,2)).reshape(b*self.heads, h*w, -1) # B*Heads,H*W,dkh
        wk = self.wk(x).view(b, h*w, -1, self.heads).permute((0,3,1,2)).reshape(b*self.heads, h*w, -1)
        wv = self.wv(x).view(b, h*w, -1, self.heads).permute((0,3,1,2)).reshape(b*self.heads, h*w, -1)
        o = F.softmax((torch.bmm(wq, wk.transpose(-1, -2))) / ((self.dk // self.heads) ** 0.5), -1)
        o = torch.bmm(o, wv)
        o = o.view(b, self.heads, h*w, -1).permute((0,2,3,1)).reshape(b, h*w, self.dv)
        o = self.wo(o)
        return o.transpose(-1, -2).view(b, -1, h, w)

    
class _AAConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.mha = _MHA2d(in_channels, int(out_channels//2))
        self.conv = nn.Conv2d(in_channels, out_channels - int(out_channels//2), 3, 1, 1, bias=bias)
        self.stride = stride
    
    def forward(self, x):
        x = torch.cat([self.mha(x), self.conv(x)], 1)
        if self.stride > 1:
            x = F.max_pool2d(x, 2)
        return x
        