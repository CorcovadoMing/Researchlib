from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


class _RandomDistill(nn.Module):
    def __init__(self, target_network, predict_network):
        super().__init__()
        self.target_network = target_network
        self.predict_network = predict_network
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
                state = torch.stack(x[i]['state'],  0).to(self.device)
                with torch.no_grad():
                    x1 = self.target_network(state)
                x2 = self.predict_network(state)
                x[i]['rnd_x'] = x1
                x[i]['rnd_y'] = x2
                with torch.no_grad():
                    intrinsic = (x1 - x2).pow(2).mean(-1).cpu()
                    intrinsic = torch.cat([intrinsic[1:], torch.zeros(1)]) # Because Ri is for s(i+1) not s(i)
                    x[i]['intrinsic'] = intrinsic
        return x
        
        