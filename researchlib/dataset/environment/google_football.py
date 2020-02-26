from .env import Env
from IPython import display
import matplotlib.pyplot as plt
import gfootball.env as football_env
import os


class GoogleFootBall(Env):
    '''
        TODO: Render, StateWrapper
        
        task = GoogleFootBall('academy_pass_and_shoot_with_keeper', 'raw', True)
        obj = task.reset()
        plt.figure(figsize=(50, 50))
        plt.imshow(obj[0][0]['frame'])
        plt.show()
        
        REF: https://github.com/google-research/football/blob/master/gfootball/doc/observation.md
    '''
    def __init__(self, name, features='raw', render=False):
        super().__init__()
        os.environ['MESA_GL_VERSION_OVERRIDE']='3.2'
        os.environ['MESA_GLSL_VERSION_OVERRIDE']='150'
        self.name = name
        self.features = features
        if self.features not in ['pixels', 'pixels_gray']:
            self._render = False
        else:
            self._render = True
        self._render = self._render | render
        self.env = football_env.create_environment(env_name=name, representation=features, render=self._render)
        self.cache = None
        self.obs = self.reset()
        print('Env', name, 'is with', self.env.observation_space, 'observed space')
        print('Env', name, 'is with', self.env.action_space, 'action space')
        
    def action_space(self):
        return self.env.action_space.n, self.env.action_space.dtype
    
    def observation_space(self):
        return self.env.observation_space.shape
    
    def samples(self):
        return self.env.action_space.sample()

    def step(self, action):
        self.obs, reward, done, info = self.env.step(action)
        return self.obs, reward, done, info

    def reset(self):
        self.obs = self.env.reset()
        return self.obs, 0, False, None

    def render(self, title = None, output = False):
        if output:
            return self.obs
        
        if self.cache is None:
            self.cache = plt.imshow(self.obs)
        else:
            self.cache.set_data(self.obs)
        
        plt.title(str(title))
        display.display(plt.gcf())
        display.clear_output(wait = True)