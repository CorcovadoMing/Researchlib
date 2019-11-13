import numpy as np
from .pack import _pack, _unpack
from ..runner import Runner
import torch
import os
from tqdm.auto import tqdm
from ..utils import Annealer
import shutil
import pickle


class Experiment:
    def __init__(self, name, runner, **kwargs):
        self.name = name
        self.runner = runner
        self.exp_key = []
        self.exp_dir = os.path.join('checkpoint', 'experiments', str(self.name))
        for key in kwargs:
            if type(kwargs[key]) != np.ndarray:
                kwargs[key] = _pack(kwargs[key])
            elif type(kwargs[key]) == np.ndarray:
                self.exp_key.append(key)
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
        
        shutil.rmtree(self.exp_dir, ignore_errors = True)
        os.makedirs(self.exp_dir, exist_ok = True)
        with open(os.path.join(self.exp_dir, 'exp_var.pkl'), 'wb') as f:
            pickle.dump(self.exp_key, f)
            
        for i in tqdm(range(self.total_runs)):
            for j in tqdm(range(times)):
                single_run_setting = {k: _unpack(v[i]) for k, v in self.settings.items()}
                if type(self.runner) == Runner:
                    Annealer.reset()
                    self.runner.start_experiment(os.path.join('experiments', str(self.name), f'runs_{i}_repeat_{j}'))
                    with open(os.path.join(self.exp_dir, f'runs_{i}_repeat_{j}', 'exp_settings.pkl'), 'wb') as f:
                        pickle.dump(single_run_setting, f)
                    self.runner.init_model('default')
                    self.runner.fit(**single_run_setting)
                else:
                    self.runner(**single_run_setting)
        