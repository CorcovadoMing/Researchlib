from .env import Env
import gym
import gym_deepmindlab
from IPython import display
import matplotlib.pyplot as plt


class DeepmindLab(Env):
    '''
        https://arxiv.org/pdf/1612.03801.pdf
    
        DeepmindLabLtChasm-v0
        DeepmindLabLtHallwaySlope-v0
        DeepmindLabLtHorseshoeColor-v0
        DeepmindLabLtSpaceBounceHard-v0
        DeepmindLabNavMazeRandomGoal01-v0
        DeepmindLabNavMazeRandomGoal02-v0
        DeepmindLabNavMazeRandomGoal03-v0
        DeepmindLabNavMazeStatic01-v0
        DeepmindLabNavMazeStatic02-v0
        DeepmindLabSeekavoidArena01-v0
        DeepmindLabStairwayToMelon-v0
        
        REF: https://github.com/jkulhanek/gym-deepmindlab-env
    '''
    def __init__(self, name, width = 224, height = 224):
        super().__init__()
        self.env = gym.make(name, width = width, height = height)
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