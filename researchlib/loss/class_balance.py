from torch import nn
import torch
from collections import Counter


class ClassBalance(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f
        if self.f.reduction != 'none':
            self.f.reduction = 'none'
        self.class_statistics = {}
        self.class_distribution = {}
    
    def forward(self, x, y):
        loss = self.f(x, y)
        
        if self.training:
            batch_statistics = dict(Counter(y.detach().cpu().numpy()))
            for i in batch_statistics:
                if i not in self.class_statistics:
                    self.class_statistics.setdefault(i, 0)
                self.class_statistics[i] += batch_statistics[i]
            
            if len(self.class_distribution):
                weighting = torch.FloatTensor([self.class_distribution[int(i)] for i in y]).to(loss.device).to(loss.dtype)
                loss = weighting * loss

        else:
            if len(self.class_statistics) > 0:
                max_ = max(self.class_statistics.values())
                for i in self.class_statistics:
                    self.class_distribution[i] = (self.class_statistics[i] / float(max_))
                self.class_statistics = {}
                
        return loss.mean()
            
        