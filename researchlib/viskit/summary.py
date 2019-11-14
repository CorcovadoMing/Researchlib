import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle


def _summary(experiment_name, xscale='linear', yscale='linear'):
    count = 0
    arr = None
    exp_dir = os.path.join('checkpoint', 'experiments', str(experiment_name))
    with open(os.path.join(exp_dir, 'exp_var.pkl'), 'rb') as f:
        exp_var = pickle.load(f)
        
    exp_var_collect = []
    
    while True:
        path = glob.glob(exp_dir + f'/runs_{count}*/*.csv')
        if not len(path):
            break
        count += 1
        
        with open(os.path.join(exp_dir, f'runs_{count-1}_repeat_0', 'exp_settings.pkl'), 'rb') as f:
            exp_settings = pickle.load(f)
        current_var = {k: exp_settings[k] for k in exp_var}
        exp_var_collect.append(current_var)
            
        name_col = pd.read_csv(path[0]).columns.values
        split_col = int((len(name_col)-1)/2)
        total = np.stack([pd.read_csv(i).values for i in path])
        
        if arr is None:
            _, arr = plt.subplots(2, split_col, figsize=(20, 10))
        
        for index in range(split_col):
            index_train = index + 1
            mean, std = total.mean(0)[:, index_train], total.std(0)[:, index_train]
            arr[index, 0].plot(range(1, total.shape[1]+1), mean)
            arr[index, 0].fill_between(range(1, total.shape[1]+1),  # plot the two-sigma uncertainty about the mean
                             (mean - 1.0 * std),
                             (mean + 1.0 * std),
                             alpha=0.3)
            arr[index, 0].set_title(f'{name_col[index_train]}')
            arr[index, 0].grid()
            arr[index, 0].set_xscale(xscale)
            arr[index, 0].set_yscale(yscale)
            if 'acc' in name_col[index_train]:
                arr[index, 0].set_yticks(np.linspace(0, 1, 11))
            
            index_val = index + split_col + 1
            mean, std = total.mean(0)[:, index_val], total.std(0)[:, index_val]
            arr[index, 1].plot(range(1, total.shape[1]+1), mean)
            arr[index, 1].fill_between(range(1, total.shape[1]+1),  # plot the two-sigma uncertainty about the mean
                             (mean - 1.0 * std),
                             (mean + 1.0 * std),
                             alpha=0.3)
            arr[index, 1].set_title(f'{name_col[index_val]}')
            arr[index, 1].grid()
            arr[index, 1].set_xscale(xscale)
            arr[index, 1].set_yscale(yscale)
            if 'acc' in name_col[index_val]:
                arr[index, 1].set_yticks(np.linspace(0, 1, 11))
    
    if not len(path) and count == 0:
        print(f'{count} experiment found in "{experiment_name}"')
    else:
        arr[0, 0].legend([f'{k}={v}' for i in range(count) for k, v in exp_var_collect[i].items()], 
                         bbox_to_anchor=(0., 1.1, 2.2, .102), 
                         ncol=count, 
                         mode="expand", 
                         borderaxespad=0)
        
        plt.show()