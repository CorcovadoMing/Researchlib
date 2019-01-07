import numpy as np
import random

class Augmentation:
    def __init__(self):
        self.aug = []
        self.aug_p = 1.0
        self.oneof = []
    
    def on(self, x, y):
        # Apply on Probability
        p_list = np.random.uniform(0, 1, len(self.aug))
        for i, func in enumerate(self.aug):
            if p_list[i] < self.aug_p:
                x, y = func(x, y)
        
        # Choose one
        func = random.choice(self.oneof)
        x, y = func(x, y)
        
        return x, y
        
            