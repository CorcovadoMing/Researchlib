import copy
import random
import numpy as np
from tqdm.auto import tqdm
from ipywidgets import IntProgress, Output, HBox, Label
from IPython import display as _display
import torch


class PBT:
    def __init__(self, runner, population=8):
        self.population = population
        self.p = [copy.deepcopy(runner) for _ in range(population)]
    
    def fit(self, pbt_epochs, unit_epochs):
        from ..preset import Preset
        
        self.lr = [1e-1 for _ in range(self.population)]
        self.wd = [1e-2 for _ in range(self.population)]
        self.m = [0.9 for _ in range(self.population)]
        
        self.epoch_progress = Output()
        self.sub_progress = Output()
        self.iteration_progress = Output()
        display(self.epoch_progress)
        display(self.sub_progress)
        display(self.iteration_progress)

        with self.epoch_progress:
            for e in tqdm(range(pbt_epochs)):
                # 1. Train
                with self.sub_progress:
                    for i in tqdm(range(len(self.p))):
                        with self.iteration_progress:
                            init = 'default' if e == 0 else None
                            print(e, self.p[i], init)
                            self.p[i].fit(**Preset.Config.Manual(epochs = unit_epochs,
                                                                 lr = self.lr[i],
                                                                 momentum = self.m[i],
                                                                 weight_decay = self.wd[i],
                                                                 expected_epochs = (pbt_epochs * unit_epochs),
                                                                 init = init, 
                                                                 use_full_liveplot = False,
                                                                 opt_info = False,
                                                                 data_info = False,
                                                                 clear_data = False,
                                                                 save_checkpoint = False,
                                                                 grad_clip = 0
                                                                ))
                            _display.clear_output()
                    _display.clear_output()

                # 2. Explore
                performance = torch.cat([torch.FloatTensor([i.last_state]) for i in self.p])
                print(performance, self.lr, self.wd, self.m)
                
                bottom_replace_candidates = performance.argsort()[:len(performance)//4]
                top_replace_candidates = performance.argsort()[-len(performance)//4:]

                for i, j in zip(bottom_replace_candidates, top_replace_candidates):
                    self.p[i].model.load_state_dict(self.p[j].model.state_dict())
                    self.p[i].val_model.load_state_dict(self.p[j].val_model.state_dict())
                    for k in range(len(self.p[j].optimizer)):
                        self.p[i].optimizer[k].load_state_dict(self.p[j].optimizer[k].state_dict())
                    self.p[i].last_state = self.p[j].last_state
                    self.lr[i] = self.lr[j]
                    self.wd[i] = self.wd[j]
                    self.m[i] = self.m[j]


                # 3. Exploit
                for i in bottom_replace_candidates:
                    self.lr[i] = self.lr[i] * np.random.uniform(0.8, 1.2)
                    self.wd[i] = self.wd[i] * np.random.uniform(0.8, 1.2)
                    self.m[i] = min(0.999, self.m[i] * np.random.uniform(0.9, 1.1))