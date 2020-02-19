from torch import nn
import random


class _ExperienceReplay(nn.Module):
    def __init__(self, total_buffer_eps=100, sample_eps=50):
        super().__init__()
        self.total_buffer_eps = total_buffer_eps
        self.sample_eps = sample_eps
        self.buffer = []
        

    def sample(self):
        # Random Sampling
        return random.choices(self.buffer, k=self.sample_eps)
        
    def forward(self, eps_trajection):
        self.buffer += eps_trajection
        if len(self.buffer) > self.total_buffer_eps:
            # FIFO
            self.buffer = self.buffer[-self.total_buffer_eps:]
        return self.sample()