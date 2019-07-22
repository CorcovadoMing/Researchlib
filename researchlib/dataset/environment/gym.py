from .env import Env
import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt

class GymEnv(Env):
    def __init__(self, name):
        super().__init__()
        self.env = gym.make(name)
        self.cache = None
        self.reset()
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self):
        self.cache = None
        return self.env.reset()
    
    def render(self, title=None):
        if self.cache is None:
            self.cache = plt.imshow(self.env.render(mode='rgb_array'))
        plt.title(str(title))
        self.cache.set_data(self.env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)