import torch
from torch import nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    '''
        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x, recon_x, mu, logvar):
        img_loss = F.mse_loss(recon_x, x)
        logvar = torch.clamp(logvar, min=-1e2, max=1e2)
        KLD = - 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        return img_loss + KLD
