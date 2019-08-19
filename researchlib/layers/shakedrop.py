import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                x,
                training=True,
                p=0.5,
                alpha_range=[-1, 1],
                beta_range=[0, 1]):
        """ Forward(Train): (gate + alpha - gate*alpha) * x
        Forward(Test):  E(gate + alpha - gate*alpha) * x
        gate is sampled from bernoulli.
        
        Args:
            training (bool): the number of current block, for example k=1 for first block
            p (int): total number of all blocks
            alpha_range (0 or list): Default: [-1, 1]
                if 0 -> drop the forward branch,
                if list -> sampled from uniform distribution with alpha_range
            beta_range (0 or list): Default: [0, 1]
                if 0 -> drop the backward branch
                if list -> sampled from uniform distribution with beta_range

        .. math::
            E(gate) = p
            E(alpha) = mean(alpha_range) = alpha_mu
            E(gate*alpha) = E(gate)*E(alpha)
            E(gate + alpha - gate*alpha) * x 
            = (E(gate) + E(alpha) - E(gate)*E(alpha)) * x
            = (p + alpha_mu - p * alpha_mu) * x
        """
        if type(alpha_range) == list:
            if len(alpha_range) != 2:
                raise ValueError("alpha_range should be two-element list or 0")
            alpha_mu = (alpha_range[0] + alpha_range[1]) / 2.0
        elif alpha_range == 0:
            alpha_mu = 0
        else:
            raise ValueError("alpha_range should be two-element list or 0")

        if type(beta_range) == list:
            if len(beta_range) != 2:
                raise ValueError("beta_range should be two-element list or 0")
        elif beta_range != 0:
            raise ValueError("beta_range should be two-element list or 0")

        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(p)
            beta_range = torch.tensor(beta_range)
            ctx.save_for_backward(gate, beta_range)
            if gate.item() == 0:
                if type(alpha_range) == list:  # two-element list
                    alpha = torch.cuda.FloatTensor(
                        x.size(0)).uniform_(*alpha_range)
                    alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                elif alpha_range == 0:
                    alpha = 0
                return alpha * x
            else:
                return x
        else:
            return (p + alpha_mu - p * alpha_mu) * x

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward: (gate + beta - gate*beta) * grad_output
        gate is sampled from bernoulli in forward().
        """
        gate, beta_range = ctx.saved_tensors
        if gate.item() == 0:
            if len(beta_range) == 2:  # two-element list
                beta = torch.cuda.FloatTensor(
                    grad_output.size(0)).uniform_(*beta_range)
                beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
                beta = Variable(beta)
            elif beta_range == 0:
                beta = beta_range
            return beta * grad_output, None, None, None, None
        else:
            return grad_output, None, None, None, None


class _ShakeDrop(nn.Module):
    """ 
    .. _ShakeDrop Regularization for Deep Residual Learning:
        https://arxiv.org/abs/1802.02375
    """

    def __init__(self,
                 block_idx,
                 block_num,
                 p=0.5,
                 alpha_range=[-1, 1],
                 beta_range=[0, 1],
                 p_L=0.5):
        """ Apply linear decay rule to compute p value for each block.
        Then p=0 for input, p=p_L for last block
        
        Note: please make sure go _ShakeDrop after BatchNorm (i.e. preact resnet will diverge easily.)
    
        Args:
            block_idx (int): the number of current block, for example k=1 for first block
            block_num (int): total number of all blocks
            p_L (float): Default: 0.5 (recommended)

        .. math::
            p = 1. - block_idx / block_num * (1.-p_L)
        
        .. _linear decay rule\: Deep Networks with Stochastic Depth:
            https://arxiv.org/abs/1603.09382
        """
        super(_ShakeDrop, self).__init__()
        self.p = 1. - block_idx / block_num * (1. - p_L)
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p,
                                       self.alpha_range, self.beta_range)
