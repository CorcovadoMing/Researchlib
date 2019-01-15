import numpy as np
import random
import torch

class Augmentation:
    def __init__(self):
        self.someof = []
        self.someof_p = 0.5
        self.oneof = []
        self.oneof_p = 0.5
    
    def on(self, x, y):
        # Apply on Probability
        if len(self.someof):
            x, y = x.numpy(), y.numpy()
            p_list = np.random.uniform(0, 1, len(self.someof))
            for i, func in enumerate(self.someof):
                if p_list[i] < self.someof_p:
                    x, y = func(x, y)

        # Choose one
        if len(self.oneof):
            func = random.choice(self.oneof)
            if np.random.uniform(0, 1, 1)[0] < self.oneof_p:
                x, y = func(x, y)

        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        
        return x, y
        
            