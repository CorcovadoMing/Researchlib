from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.init as init
import copy


class _RandomNetworkDistillation(nn.Module):
    def __init__(self, predict_network, target_network = None):
        super().__init__()
        self.target_network = target_network
        self.predict_network = predict_network
        if self.target_network is None:
            self.target_network = copy.deepcopy(self.predict_network)
        self.device = None
        for i in self.target_network.parameters():
            i.requires_grad = False
            if i.ndim > 1:
                init.orthogonal_(i)
        
    def forward(self, x):
        if self.device is None:
            self.device = next(self.predict_network.parameters()).device
        
        if type(x) == list: # RL
            for i in range(len(x)):
                for j in range(len(x[i])):
                    state = torch.stack(x[i][j]['state'],  0).to(self.device)
                    with torch.no_grad():
                        x1 = self.target_network(state).detach()
                    x2 = self.predict_network(state)
                    x[i][j]['intrinsic_loss'] = (x2 - x1).abs().sum(-1)
                    with torch.no_grad():
                        intrinsic = (x2 - x1).abs().sum(-1).detach().cpu()
                        intrinsic = torch.cat([intrinsic[1:], torch.zeros(1)]) # Because Ri is for s(i+1) not s(i)
                        x[i][j]['intrinsic'] = intrinsic
            return x
        
        else: # Normal flow
            return self.predict_network(x), self.target_network(x)
        
        