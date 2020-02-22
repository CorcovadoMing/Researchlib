from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from .env import Env
from IPython import display
import matplotlib.pyplot as plt


class SuperMarioBros(Env):
    '''
        REF: https://github.com/Kautenja/gym-super-mario-bros
    '''
    def __init__(self, name):
        super().__init__()
        env = gym_super_mario_bros.make(name)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.env = env
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

    def render(self, title = None):
        if self.cache is None:
            self.cache = plt.imshow(self.env.render(mode = 'rgb_array'))
        plt.title(str(title))
        self.cache.set_data(self.env.render(mode = 'rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait = True)