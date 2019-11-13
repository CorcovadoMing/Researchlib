import numpy as np
from .pack import _pack, _unpack
from ..runner import Runner
import torch
import os
from tqdm.auto import tqdm
from ..utils import Annealer
    
class Experiment:
    def __init__(self, runner, **kwargs):
        self.runner = runner
        for key in kwargs:
            if type(kwargs[key]) != np.ndarray:
                kwargs[key] = _pack(kwargs[key])
        self.kwargs = kwargs
        total_key = list(kwargs.keys())
        total_values = list(kwargs.values())
        all_combination = np.meshgrid(*total_values)
        all_combination = [i.flatten() for i in all_combination]
        self.settings = {key: value for key, value in zip(total_key, all_combination)}
        self.total_runs = len(all_combination[0])
    
    def repeat(self, times, seed=0):
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        
        for i in tqdm(range(self.total_runs)):
            for j in tqdm(range(times)):
                single_run_setting = {k: _unpack(v[i]) for k, v in self.settings.items()}
                if type(self.runner) == Runner:
                    Annealer.reset()
                    self.runner.start_experiment(os.path.join('experiments', f'runs_{i}_repeat_{j}'))
                    self.runner.init_model('default')
                    self.runner.fit(**single_run_setting)
                else:
                    self.runner(**single_run_setting)
        