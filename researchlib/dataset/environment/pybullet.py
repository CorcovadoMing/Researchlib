from .env import Env
import gym
import pybulletgym
from IPython import display
import matplotlib.pyplot as plt
import warnings


class PyBulletEnv(Env):
    def __init__(self, name):
        super().__init__()
        warnings.filterwarnings('ignore')
        self.env = gym.make(name)
        self.cache = None
        self.reset()
        print('Env', name, 'is with', self.env.observation_space, 'observed space')
        print('Env', name, 'is with', self.env.action_space, 'action space')
        warnings.filterwarnings('once')
        
    def action_space(self):
        return self.env.action_space.n, self.env.action_space.dtype
    
    def observation_space(self):
        return self.env.observation_space.shape
    
    def samples(self):
        return self.env.action_space.sample()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.cache = None
        return self.env.reset(), 0, False, None

    def render(self, title = None, output = False):
        if output:
            return self.env.render(mode = 'rgb_array')
        
        if self.cache is None:
            self.cache = plt.imshow(self.env.render(mode = 'rgb_array'))
        else:
            self.cache.set_data(self.env.render(mode = 'rgb_array'))
        
        plt.title(str(title))
        display.display(plt.gcf())
        display.clear_output(wait = True)