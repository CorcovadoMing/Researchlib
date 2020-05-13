from torch import nn
import torch


class IMSAT(nn.Module):
    def __init__(self):
        super().__init__()
    
    def entropy(self, p):
        p = torch.clamp(p, min=1e-6, max=1)
        return - (p * p.log()).sum(-1).mean()
        
    def compute_entropy(self, p):
        p_avg = p.mean(0)
        return self.entropy(p), self.entropy(p_avg)
        
    def kl(self, p, q):
        p = torch.clamp(p, min=1e-6, max=1)
        q = torch.clamp(q, min=1e-6, max=1)
        return - (p * q.log()).sum(-1).mean()
    
    def forward(self, p, q):
        etp, avg_etp = self.compute_entropy(p)
        p = p.detach()
        rsat = self.kl(p, q)
        return rsat + 0.5 * etp - avg_etp