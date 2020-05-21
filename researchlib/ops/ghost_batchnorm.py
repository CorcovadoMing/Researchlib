import torch
from torch import nn


class _GhostBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, effective_batchsize=32, **kw):
        super().__init__(num_features, **kw)
        self.effective_batchsize = effective_batchsize
        self.num_splits = 1
        self.num_features = num_features
        
    def __allocate_statistics(self, cur_batch):
        self.num_splits = cur_batch // self.effective_batchsize
        self.register_buffer('running_mean', torch.mean(self.running_mean.view(-1, self.num_features), dim=0).repeat(self.num_splits))
        self.register_buffer('running_var', torch.mean(self.running_var.view(-1, self.num_features), dim=0).repeat(self.num_splits))
        
    def train(self, mode=True):
        if (self.training is True) and (mode is False): #lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(self.num_splits)
        return super().train(mode)
        
    def forward(self, input):
        N, C, H, W = input.shape
        
        if self.training or not self.track_running_stats:
            num_splits = N // self.effective_batchsize
            if num_splits != self.num_splits:
                self.__allocate_statistics(N)
                
            return nn.functional.batch_norm(
                input.view(-1, C*num_splits, H, W), self.running_mean, self.running_var, 
                self.weight.repeat(num_splits), self.bias.repeat(num_splits),
                True, self.momentum, self.eps).view(N, C, H, W) 
        else:
            
            return nn.functional.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features], 
                self.weight, self.bias, False, self.momentum, self.eps)