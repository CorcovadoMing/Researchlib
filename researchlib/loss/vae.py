import torch
from torch import nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, arch = 'vanilla'):
        super().__init__()
        self.arch = arch

    def forward(self, out, original):
        reconstruction, mu, logvar = out
        recons_loss = F.mse_loss(reconstruction, original)
        kl_distance = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recons_loss + kl_distance
