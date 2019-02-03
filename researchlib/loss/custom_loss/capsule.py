from ...utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, size_average=True, loss_lambda=0.5):
        '''
        Margin loss for digit existence
        Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

        Args:
            size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
            loss_lambda: parameter for down-weighting the loss for missing digits
        '''
        super().__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda

    def forward(self, inputs, labels):
        inputs = torch.norm(inputs, dim=-1).cuda()
        L_k = labels * F.relu(self.m_plus - inputs)**2 + self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
        L_k = L_k.sum(dim=1)

        if self.size_average:
            return L_k.mean()
        else:
            return L_k.sum()

class CapsuleLoss(nn.Module):
    def __init__(self, loss_lambda=0.5, recon_loss_scale=5e-4, size_average=True):
        '''
        Combined margin loss and reconstruction loss. Margin loss see above.
        Sum squared error (SSE) was used as a reconstruction loss.

        Args:
            recon_loss_scale: 	param for scaling down the the reconstruction loss
            size_average:		if True, reconstruction loss becomes MSE instead of SSE
        '''
        super().__init__()
        self.size_average = size_average
        self.margin_loss = MarginLoss(size_average=size_average, loss_lambda=loss_lambda)
        self.reconstruction_loss = nn.MSELoss(reduction='mean')
        self.recon_loss_scale = recon_loss_scale
        self.require_data = True
        
    def forward(self, inputs, labels, images):
        reconstructions = inputs[1]
        inputs = inputs[0]
        labels = to_one_hot(labels, 10).cuda()
        margin_loss = self.margin_loss(inputs, labels)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)
        caps_loss = (margin_loss + self.recon_loss_scale * reconstruction_loss)

        return caps_loss
