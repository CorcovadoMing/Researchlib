from torch import nn
import robust_loss_pytorch
import numpy as np


class AdaptiveRobustLoss(nn.Module):

    def __init__(self, num_dims):
        super().__init__()
        self.adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=num_dims, float_dtype=np.float32, device='cuda:0')

    def forward(self, x, y):
        return self.adaptive_loss.lossfun(x - y).mean()

    def alpha(self):
        return self.adaptive_loss.alpha()

    def scale(self):
        return self.adaptive_loss.scale()
