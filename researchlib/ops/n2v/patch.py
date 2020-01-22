from torch import nn
import torch.nn.functional as F


class _Patch(nn.Module):
    def __init__(self, patch_size, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
    
    def forward(self, x):
        if self.training:
            x = F.unfold(x, self.patch_size, 1, 0, self.stride).reshape(x.size(0), self.patch_size, self.patch_size, -1).permute(0, 3, 1, 2)
            x = x.reshape(-1, 1, x.size(-2), x.size(-1))
        return x