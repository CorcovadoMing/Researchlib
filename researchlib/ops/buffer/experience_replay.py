from torch import nn
import numpy as np


class _ExperienceReplay(nn.Module):
    def __init__(self, total_len=64, samples_poisson=8, samples_len=-1):
        super().__init__()
        self.total_len = total_len
        self.samples_len = samples_len
        self.samples_poisson = samples_poisson
        self.samples_len_type = 'static' if samples_len >= 0 else 'dynamic'
        self.buffer = []
        self.cache = None
        self.enable = True

    def set_enable(self):
        self.enable = True
        
    def set_disable(self):
        self.enable = False
        
    def samples(self):
        if self.samples_len_type == 'dynamic':
            n = np.random.poisson(self.samples_poisson)
        else:
            n = self.samples_len
        return self.buffer[-(1+n):] # on-policy + n * off-policy
        
    def forward(self, eps_trajection):
        if self.enable:
            self.buffer.append(eps_trajection)
            if len(self.buffer) > self.total_len:
                self.buffer = self.buffer[-self.total_len:]
            self.cache = self.samples()
        return self.cache