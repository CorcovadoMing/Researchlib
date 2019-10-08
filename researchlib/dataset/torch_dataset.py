import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from functools import partial


class _TorchDataset:
    def __init__(self, name, is_train):
        self.is_train = is_train
        self.name = name

        dataset_fn = None
        for i in torchvision.datasets.__dict__:
            if i.lower() == name:
                dataset_fn = torchvision.datasets.__dict__[i]
                break
        
        if dataset_fn is None:
            raise ValueError(f'No dataset {name} founded')
            
        self.dataset_fn = partial(dataset_fn, train=is_train, download=True, root='./data')
        
        self.normalizer = []
        self.augmentor = []
        self.include_y = False
        
        
    def _set_normalizer(self, local=False):
        if not local:
            self.normalizer = [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        else:
            self.normalizer = [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    
    
    def _set_augmentor(self, augmentor, include_y=False):
        mapping = {
            'hflip': transforms.RandomHorizontalFlip(),
            'crop': transforms.RandomCrop(32, padding=4)
        }
        _aug = []
        for i in augmentor:
            _aug.append(mapping[i])
        self.augmentor = _aug
        self.include_y = include_y
    
    def get_generator(self, batch_size=512, **kwargs):
        if self.is_train:
            tf = self.augmentor + [transforms.ToTensor()] + self.normalizer
        else:
            tf = [transforms.ToTensor()] + self.normalizer
        tf = transforms.Compose(tf)
        ds = self.dataset_fn(transform=tf)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=self.is_train, num_workers=4)