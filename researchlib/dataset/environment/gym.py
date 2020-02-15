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
        print('Env', name, 'is with', self.env.observation_space, 'observed space')
        print('Env', name, 'is with', self.env.action_space, 'action space')
        
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

    def render(self, title = None, return_cache = False):
        if return_cache:
            return self.env.render(mode = 'rgb_array')
        else:
            if self.cache is None:
                self.cache = plt.imshow(self.env.render(mode = 'rgb_array'))
            plt.title(str(title))
            self.cache.set_data(self.env.render(mode = 'rgb_array'))
            display.display(plt.gcf())
            display.clear_output(wait = True)
