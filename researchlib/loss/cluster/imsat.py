from torch import nn
import torch


class IMSAT(nn.Module):
    def __init__(self):
        super().__init__()
    
    def entropy(self, p):
        # compute entropy
        if (len(p.size())) == 2:
            return - torch.sum(p * torch.log(p + 1e-2)) / float(len(p))
        elif (len(p.size())) == 1:
            return - torch.sum(p * torch.log(p + 1e-2))
        else:
            raise NotImplementedError
        
    def compute_entropy(self, p):
        p_avg = p.mean(0)
        return self.entropy(p), self.entropy(p_avg)
        
    def kl(self, p, q):
        return torch.sum(p * torch.log((p / (q + 1e-2)) + 1e-2)) / float(len(p))
    
    def forward(self, p, q):
        avg_etp, etp = self.compute_entropy(p)
        p = p.detach()
        rsat = self.kl(p, q)
        return rsat + 0.5 * (avg_etp - 4. * etp)