import pandas as pd
import glob
import matplotlib.pyplot as plt
import os
import numpy as np


def _summary(experiment_name):
    count = 0
    arr = None
    
    while True:
        path = glob.glob(os.path.join('checkpoint', 'experiments', str(experiment_name)) + f'/runs_{count}*/*.csv')
        if not len(path):
            break
        count += 1
        
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
                             (mean - 3.0 * std),
                             (mean + 3.0 * std),
                             alpha=0.3)
            arr[index, 0].set_title(f'{name_col[index_train]}')
            
            index_val = index + split_col + 1
            mean, std = total.mean(0)[:, index_val], total.std(0)[:, index_val]
            arr[index, 1].plot(range(1, total.shape[1]+1), mean)
            arr[index, 1].fill_between(range(1, total.shape[1]+1),  # plot the two-sigma uncertainty about the mean
                             (mean - 3.0 * std),
                             (mean + 3.0 * std),
                             alpha=0.3)
            arr[index, 1].set_title(f'{name_col[index_val]}')
    
    if not len(path) and count == 0:
        print(f'{count} experiment found in "{experiment_name}"')
    else:
        arr[0, 0].legend([f'Experiments {i}' for i in range(1, count+1)], bbox_to_anchor=(0., 1.1, 2.2, .102), ncol=count, mode="expand", borderaxespad=0)  
        plt.show()