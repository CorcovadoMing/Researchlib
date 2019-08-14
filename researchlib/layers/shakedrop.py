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
        """ gate is sampled from bernoulli.
        Forward(Train): (gate + alpha - gate*alpha) * x
        Forward(Test):  E(gate + alpha - gate*alpha) * x

        E(gate) = p
        E(alpha) = mean(alpha_range) = alpha_mu
        E(gate*alpha) = E(gate)*E(alpha)
        E(gate + alpha - gate*alpha) * x 
        = (E(gate) + E(alpha) - E(gate)*E(alpha)) * x
        = (p + alpha_mu - p * alpha_mu) * x

        """
        alpha_mu = (alpha_range[0] + alpha_range[1]) / 2.0
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(p)
            beta_range = torch.tensor(beta_range)
            ctx.save_for_backward(gate, beta_range)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(
                    x.size(0)).uniform_(*alpha_range)
                alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (p + alpha_mu - p * alpha_mu) * x

    @staticmethod
    def backward(ctx, grad_output):
        """ gate is sampled from bernoulli in forward().
        Backward: (gate + beta - gate*beta) * grad_output
        """
        gate, beta_range = ctx.saved_tensors
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(
                grad_output.size(0)).uniform_(*beta_range)
            beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
            beta = Variable(beta)
            return beta * grad_output, None, None, None, None
        else:
            return grad_output, None, None, None, None


class _ShakeDrop(nn.Module):
    """ Then p=0 for input, p=p_L for last block
    
    Args:
        block_idx (int): the number of current block, for example k=1 for first block
        block_num (int): total number of all blocks
        p_L (float): Default: 0.5 (recommended)
    
    .. math::
        p = 1. - block_idx / block_num * (1.-p_L)
    
    .. _linear decay rule\: Deep Networks with Stochastic Depth:
        https://arxiv.org/abs/1603.09382
    """
    def __init__(self,
                 block_idx,
                 block_num,
                 p=0.5,
                 alpha_range=[-1, 1],
                 beta_range=[0, 1],
                 p_L=0.5):
        super(_ShakeDrop, self).__init__()
        self.p = 1. - block_idx / block_num * (1. - p_L)
        self.alpha_range = alpha_range
        self.beta_range = beta_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p,
                                       self.alpha_range, self.beta_range)
