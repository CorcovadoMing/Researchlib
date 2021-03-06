import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training, p, alpha_range, beta_range, mode):
        '''
        Forward(Train): (gate + alpha - gate*alpha) * x
        Forward(Test):  E(gate + alpha - gate*alpha) * x
        gate is sampled from bernoulli.
        
        Args:
            p (float): the percentages to be kept.

        .. math::
            E(gate) = p
            E(alpha) = mean(alpha_range) = alpha_mu
            E(gate*alpha) = E(gate)*E(alpha)
            E(gate + alpha - gate*alpha) * x 
            = (E(gate) + E(alpha) - E(gate)*E(alpha)) * x
            = (p + alpha_mu - p * alpha_mu) * x
        '''

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
            gate = torch.cuda.FloatTensor([0]).to(x.dtype).bernoulli_(p)
            beta_range = torch.tensor(beta_range)
            ctx.save_for_backward(mode, gate, beta_range)
            if gate.item() == 0:
                if type(alpha_range) == list:  # two-element list
                    if int(mode) == 0:
                        alpha = torch.FloatTensor(1).to(x.device).to(x.dtype
                                                                     ).uniform_(*alpha_range)
                        new_shape = []
                        while len(new_shape) != x.dim():
                            new_shape.append(1)
                        alpha = alpha.view(*new_shape).expand_as(x)
                    if int(mode) == 1:
                        alpha = torch.FloatTensor(x.size(0)).to(x.device
                                                                ).to(x.dtype
                                                                     ).uniform_(*alpha_range)
                        new_shape = [
                            alpha.size(0),
                        ]
                        while len(new_shape) != x.dim():
                            new_shape.append(1)
                        alpha = alpha.view(*new_shape).expand_as(x)
                    elif int(mode) == 2:
                        alpha = torch.FloatTensor(x.size(0),
                                                  x.size(1)).to(x.device
                                                                ).to(x.dtype
                                                                     ).uniform_(*alpha_range)
                        new_shape = [alpha.size(0), alpha.size(1)]
                        while len(new_shape) != x.dim():
                            new_shape.append(1)
                        alpha = alpha.view(*new_shape).expand_as(x)
                    elif int(mode) == 3:
                        alpha = torch.empty_like(x).to(x.device).to(x.dtype).uniform_(*alpha_range)
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
        mode, gate, beta_range = ctx.saved_tensors
        if gate.item() == 0:
            if len(beta_range) == 2:  # two-element list
                if int(mode) == 0:
                    beta = torch.FloatTensor(1).to(grad_output.device).to(grad_output.dtype
                                                                          ).uniform_(*beta_range)
                    new_shape = []
                    while len(new_shape) != grad_output.dim():
                        new_shape.append(1)
                    beta = beta.view(*new_shape).expand_as(grad_output)
                if int(mode) == 1:
                    beta = torch.FloatTensor(grad_output.size(0)).to(grad_output.device
                                                                     ).to(grad_output.dtype
                                                                          ).uniform_(*beta_range)
                    new_shape = [
                        beta.size(0),
                    ]
                    while len(new_shape) != grad_output.dim():
                        new_shape.append(1)
                    beta = beta.view(*new_shape).expand_as(grad_output)
                elif int(mode) == 2:
                    beta = torch.FloatTensor(grad_output.size(0),
                                             grad_output.size(1)).to(grad_output.device
                                                                     ).to(grad_output.dtype
                                                                          ).uniform_(*beta_range)
                    new_shape = [beta.size(0), beta.size(1)]
                    while len(new_shape) != grad_output.dim():
                        new_shape.append(1)
                    beta = beta.view(*new_shape).expand_as(grad_output)
                elif int(mode) == 3:
                    beta = torch.empty_like(grad_output).to(grad_output.device
                                                            ).to(grad_output.dtype
                                                                 ).uniform_(*beta_range)
                beta = Variable(beta)
            elif beta_range == 0:
                beta = 0
            return beta * grad_output, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None


class _ShakeDrop(nn.Module):
    '''
        ShakeDrop Regularization for Deep Residual Learning:
        https://arxiv.org/abs/1802.02375
    '''
    def __init__(
        self,
        block_idx,
        block_num,
        alpha_range = [-1, 1],
        beta_range = [0, 1],
        shakedrop_prob = 0.5,
        mode = 3
    ):
        '''
        Apply linear decay rule to compute p value for each block.
        Then p=1 for input, p=p_L for last block
        
        Note: please make sure go _ShakeDrop after BatchNorm (i.e. preact resnet will diverge easily.)
    
        Args:
            block_idx (int): the number of current block, for example k=1 for first block
            block_num (int): total number of all blocks
            alpha_range (0 or list): Default: [-1, 1]
                if 0 -> drop the forward branch,
                if list -> sampled from uniform distribution with alpha_range
            beta_range (0 or list): Default: [0, 1]
                if 0 -> drop the backward branch
                if list -> sampled from uniform distribution with beta_range
            shakedrop_prob (float): Default: 0.5, represent (1. - p_L) in paper.
                pL = 0.025, 0.05 recommended for small model (< 50 layers) on large dataset (Imagenet)
                pL = 0.5 recommended for complex model (RandomDrop) and small dataset (Cifar)
            mode: 0, 1, 2, 3 for batch, sample, channel, pixel mode
                mode = 3, pixel level is recommended in paper

        .. math::
            p = 1. - block_idx / block_num * (1.-p_L)
        
        .. _linear decay rule\: Deep Networks with Stochastic Depth:
            https://arxiv.org/abs/1603.09382
        '''

        super().__init__()
        self.p = 1. - ((block_idx / block_num) * shakedrop_prob)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.mode = torch.empty(1).fill_(mode)

    def forward(self, x):
        return ShakeDropFunction.apply(
            x, self.training, self.p, self.alpha_range, self.beta_range, self.mode
        )
