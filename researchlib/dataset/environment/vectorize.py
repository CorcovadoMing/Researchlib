from IPython import display
import numpy as np
import matplotlib.pyplot as plt


class Vectorize:
    def __init__(self, env_list):
        self.env_list = env_list
    
    def reset(self):
        _s = []
        _o = []
        _d = []
        _info = []
        for env in self.env_list:
            s, o, d, info = env.reset()
            _s.append(s)
            _o.append(o)
            _d.append(d)
            _info.append(info)
        return np.array(_s), np.array(_o), np.array(_d), np.array(_info)
    
    def action_space(self):
        return self.env_list[0].action_sapce()
    
    def observation_space(self):
        return self.env_list[0].observation_space()
    
    def samples(self):
        return [i.samples() for i in self.env_list]
    
    def step(self, action):
        _s = []
        _o = []
        _d = []
        _info = []
        for env, act in zip(self.env_list, action):
            s, o, d, info = env.step(act)
            _s.append(s)
            _o.append(o)
            _d.append(d)
            _info.append(info)
        return np.array(_s), np.array(_o), np.array(_d), np.array(_info)
    
    def render(self, title=None):
        screen = []
        for env in self.env_list:
            screen.append(env.render(output=True))
        screen = np.concatenate(screen, 1)
        plt.figure(figsize=(20, 20))
        plt.title(str(title))
        plt.imshow(screen)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        display.clear_output(wait = True)
        