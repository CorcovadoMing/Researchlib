import torch
import torchvision
import numpy as np
from functools import partial
from tensorpack.dataflow import *
from .process_single import _process_single
from .augmentations import augmentations
from .preprocessing import preprocessing


class ToNumpy:
    def __call__(self, img):
        return np.array(img)

def my_collate(batch):
    x = [i[0] for i in batch]
    y = [i[1] for i in batch]
    return x, y


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
            
        self.ds = dataset_fn(train=is_train, download=True, root='./data', transform=ToNumpy())
        
        self.normalizer = []
        self.augmentor = []
        self.include_y = False
        
        
    def set_normalizer(self, type, mean, std):
        self.normalizer = preprocessing.set_normalizer(type, mean, std)
    
    
    def _set_augmentor(self, augmentor, include_y=False):
        mapping = {
            'hflip': augmentations.HFlip(),
            'crop': augmentations.Crop(32, 32, 4)
        }
        
        self.augmentor = []
        for i in augmentor:
            self.augmentor.append(mapping[i])
        self.include_y = include_y
    
    def get_generator(self, batch_size=512, **kwargs):
        ds = torch.utils.data.DataLoader(self.ds, 
                                         batch_size=batch_size, 
                                         shuffle=self.is_train, 
                                         collate_fn=my_collate,
                                         num_workers=1)
        length = len(ds)
        ds = DataFromGenerator(ds)
        ds.__len__ = lambda: length
        process_single_fn = partial(_process_single, 
                                    is_train=self.is_train, include_y=self.include_y, 
                                    normalizer=self.normalizer, augmentor=self.augmentor)
        ds = MultiProcessMapDataZMQ(ds, 4, process_single_fn)
        ds = PrintData(ds)
        ds.reset_state()
        return ds