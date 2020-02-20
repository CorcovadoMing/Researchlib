from torch import nn
import random


class _ExperienceReplay(nn.Module):
    def __init__(self, total_buffer_eps=2000, sample_eps=1000):
        super().__init__()
        self.total_buffer_eps = total_buffer_eps
        self.sample_eps = sample_eps
        self.buffer = []
        self.cache = None
        self.enable = True

    def set_enable(self):
        self.enable = True
        
    def set_disable(self):
        self.enable = False
        
    def sample(self):
        # Random Sampling
        return random.choices(self.buffer, k=self.sample_eps)
        
    def forward(self, eps_trajection):
        if self.enable:
            self.buffer += eps_trajection
            if len(self.buffer) > self.total_buffer_eps:
                self.buffer = self.buffer[-self.total_buffer_eps:]
            self.cache = self.sample()
        return self.cache