import torch
from torch import nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    '''
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
        0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, recon_x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x.view(recon_x.size(0), -1), x.view(x.size(0), -1), reduction = 'sum') / x.size(0)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return BCE + KLD
