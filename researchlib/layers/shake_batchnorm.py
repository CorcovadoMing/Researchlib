from torch import nn
import torch


class _ShakeBatchNorm1d(nn.Module): 

    def __init__(self, num_features):
        raise NotImplementedError

class _ShakeBatchNorm2d(nn.Module):

    def __init__(self, num_features, bn_affine, gamma_range, beta_range):
        super().__init__()
        self.num_features = num_features
        self.bn_affine = bn_affine
        self.gamma_range = gamma_range
        self.beta_range = beta_range
        self.bn_layer = nn.BatchNorm2d(self.num_features, affine=self.bn_affine)

    def forward(self, x):
        """
        gamma_range (TODO)
        """
        x = self.bn_layer(x)
        if self.training:
            gamma = self._sample(self.gamma_range, x)
            beta = self._sample(self.beta_range, x)
        else:
            gamma = (self.gamma_range[0] + self.gamma_range[1]) / 2.0
            beta = (self.beta_range[0] + self.beta_range[1]) / 2.0
        return x * gamma + beta
    
    def _sample(self, range_, x):
        if range_[0] == range_[1]:
            samples = range_[0]
        else:
            samples = torch.FloatTensor(self.num_features).to(x.device).to(x.dtype).uniform_(*range_)
            new_shape = [1, x.size(1)]
            while len(new_shape) != x.dim(): 
                new_shape.append(1)
            samples = samples.view(*new_shape).expand_as(x)
        return samples
        

class _ShakeBatchNorm3d(nn.Module):

    def __init__(self, num_features):
        raise NotImplementedError