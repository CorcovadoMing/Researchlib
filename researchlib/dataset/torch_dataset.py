import torchvision
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
from .np_dataset import _NumpyDataset


def _TorchDataset(name, is_train, shuffle, noise_ratio=0, noise_type='asymmetry', label_mapping=None):
    noise_type_candidates = ['asymmetry', 'symmetry']
    if noise_type not in noise_type_candidates:
        raise ValueError(f'Noise type can only be one of the {noise_type_candidates}')
    
    dataset_fn = None
    for i in torchvision.datasets.__dict__:
        if i.lower() == name and type(torchvision.datasets.__dict__[i]) == type:
            dataset_fn = torchvision.datasets.__dict__[i]
            break

    if dataset_fn is None:
        raise ValueError(f'No dataset {name} founded')

    ds = dataset_fn(train = is_train, download = True, root = './data')
    
    if name == 'mnist':
        ds.data = ds.data.unsqueeze(-1)
    
    if type(ds.data) == torch.Tensor:
        data = ds.data.numpy().astype(np.float32)
        target = ds.targets.numpy() 
    else:
        data = np.array(ds.data).astype(np.float32)
        target = np.array(ds.targets)
    
    id_list = np.arange(len(target)).astype(np.int64)
    total_class = target.max() + 1
    original_target = copy.deepcopy(target)
    
    if noise_ratio != 0:
        print(f'Apply label noise type: {noise_type}')
        print(f'Total classes: {total_class}')
        
        if noise_type == 'asymmetry':
            noise_idx = list(range(len(target)))
            random.shuffle(noise_idx)
            noise_idx = noise_idx[:int(len(target)*noise_ratio)]
            target[noise_idx] = (target[noise_idx] + 1) % total_class
        else:
            noise_idx = list(range(len(target)))
            random.shuffle(noise_idx)
            noise_idx = noise_idx[:int(len(target)*noise_ratio)]
            for i in noise_idx:
                target[i] = np.random.choice(np.delete(np.arange(total_class), target[i]))
        
        confusion_matrix = np.zeros((total_class, total_class), dtype=np.int64)
        for i in range(len(target)):
            confusion_matrix[target[i], original_target[i]] += 1
        print(confusion_matrix)
    
    if label_mapping is not None:
        count = 0
        print(f'Load label mapping {label_mapping} as the label ...')
        mapping = torch.load(label_mapping)
        shape = list(mapping.values())[0].shape
        new_target = []
        for i in id_list:
            if i in mapping:
                new_target.append(mapping[i].numpy())
            else:
                count += 1
                target_i = np.array(target[i])
                if target_i.shape != shape:
                    arr = np.zeros([total_class])
                    arr[target_i] = 1
                    new_target.append(arr)
                else:
                    new_target.append(target_i)
        target = np.stack(new_target)
        print(f'Done. Missing {count} labels')
            
    return _NumpyDataset(data, target, original_target, id_list, shuffle = shuffle, name = name)