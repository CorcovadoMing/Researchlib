from torch import nn
import torch


class _DropReLU(nn.Module):
    '''
        Drop-Activation: Implicit Parameter Reduction and Harmonious Regularization
        https://arxiv.org/abs/1811.05850
    '''
    def __init__(self, keep_prob = 0.95):
        '''
        :param keep_prob: the probability of retaining the ReLU activation
        '''
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, x):
        '''
        :param x: input of x
        :return: drop activation during training or testing phase
        '''
        size_len = len(x.size())
        if self.training:
            Bernoulli_mask = torch.cuda.FloatTensor(x.size()[0:size_len]).fill_(1)
            Bernoulli_mask.bernoulli_(self.keep_prob)
            temp = torch.Tensor().cuda()
            output = torch.Tensor().cuda()
            temp.resize_as_(x).copy_(x)
            output.resize_as_(x).copy_(x)
            output.mul_(Bernoulli_mask)
            output.mul_(-1)
            output.add_(temp)
            temp.clamp_(min = 0)
            temp.mul_(Bernoulli_mask)
            output.add_(temp)
            return output
        else:
            temp = torch.Tensor().cuda()
            output = torch.Tensor().cuda()
            temp.resize_as_(x).copy_(x)
            output.resize_as_(x).copy_(x)
            output.mul_(self.keep_prob)
            output.mul_(-1)
            output.add_(temp)
            temp.clamp_(min = 0)
            temp.mul_(self.keep_prob)
            output.add_(temp)
            return output
