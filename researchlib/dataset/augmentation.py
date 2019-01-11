import numpy as np
import random
import torch

class Augmentation:
    def __init__(self):
        self.aug = []
        self.aug_p = 0.5
        self.oneof = []
        self.oneof_p = 0.5
    
    def on(self, x, y):
        # Apply on Probability
        x, y = x.numpy(), y.numpy()
        p_list = np.random.uniform(0, 1, len(self.aug))
        for i, func in enumerate(self.aug):
            if p_list[i] < self.aug_p:
                x, y = func(x, y)
        
        # Choose one
        func = random.choice(self.oneof)
        if np.random.uniform(0, 1, 1)[0] < self.oneof_p:
            x, y = func(x, y)
        
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        return x, y
        
            