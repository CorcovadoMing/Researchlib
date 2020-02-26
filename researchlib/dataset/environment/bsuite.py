from .env import Env
from IPython import display
import matplotlib.pyplot as plt
import bsuite
import numpy as np


class BSuiteEnv(Env):
    def __init__(self, name):
        super().__init__()
        self.env = bsuite.load_from_id(name)
        self.cache = None
        self.reset()
        print('Env', name, 'is with', self.env.observation_spec(), 'observed space')
        print('Env', name, 'is with', self.env.action_spec(), 'action space')
        
    def action_space(self):
        return self.env.action_spec().num_values, self.env.action_spec().dtype
    
    def observation_space(self):
        return self.env.observation_spec().shape
    
    def samples(self):
        n, d = self.action_space()
        return np.random.choice(n)

    def step(self, action):
        obj = self.env.step(action)
        return obj.observation, obj.reward, obj.last(), None

    def reset(self):
        self.cache = None
        obj = self.env.reset()
        return obj.observation, obj.reward, obj.last(), None
        

    def render(self, title = None, output = False):
        pass