import numpy as np
import random
import torch

class Augmentator:
    def __init__(self):
        self.someof = []
        self.someof_p = 0.5
        self.oneof = []
        self.oneof_p = 0.5
    
    def on(self, x, y):
        x = [i.numpy() for i in x]
        y = [i.numpy() for i in y]
        
        # Apply on Probability        
        if len(self.someof):
            p_list = np.random.uniform(0, 1, len(self.someof))
            for i, func in enumerate(self.someof):
                if p_list[i] < self.someof_p:
                    for ind in range(len(x)):
                        x[ind], y[ind] = func(x[ind], y[ind])

        # Choose one
        if len(self.oneof):
            func = random.choice(self.oneof)
            if np.random.uniform(0, 1, 1)[0] < self.oneof_p:
                for ind in range(len(x)):
                    x[ind], y[ind] = func(x[ind], y[ind])

        x = [torch.from_numpy(np.ascontiguousarray(i)).float() for i in x]
        y = [torch.from_numpy(np.ascontiguousarray(i)).float() for i in y]
        
        return x, y
        
            