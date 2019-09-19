from torch import nn
import torch
import torch.nn.functional as F


class _ShakeBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        raise NotImplementedError


class _ShakeBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.mu = nn.Parameter(torch.randn(1, num_features, 1, 1))
        self.logvar = nn.Parameter(torch.randn(1, num_features, 1, 1))

    def forward(self, x):
        # Instance norm, no batch size effect
        original_mean = x.mean(dim = [2, 3])[:, :, None, None]
        x = x - original_mean
        original_std = x.std(dim = [2, 3])[:, :, None, None]
        x = x / original_std
        return x + self.reparameterize()

    def reparameterize(self):
        std = torch.exp(0.5 * self.logvar)
        std = F.softplus(std) + 1e-4
        eps = torch.randn_like(std)
        return self.mu + eps * std


class _ShakeBatchNorm3d(nn.Module):
    def __init__(self, num_features):
        raise NotImplementedError
