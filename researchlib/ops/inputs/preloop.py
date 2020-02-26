from tensorpack.dataflow import *
from torch import nn
import numpy as np
import torch
from ...utils import ParameterManager

def cov(X):
    X = X/np.sqrt(X.size(0) - 1)
    return X.t() @ X

def patches(data, patch_size=(3, 3), dtype=torch.float32):
    h, w = patch_size
    c = data.size(1)
    return data.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1, c, h, w).to(dtype)

def eigens(patches):
    n,c,h,w = patches.shape
    Σ = cov(patches.reshape(n, c*h*w))
    Λ, V = torch.symeig(Σ, eigenvectors=True)
    return Λ.flip(0), V.t().reshape(c*h*w, c, h, w).flip(0)


class collect_statistics:
    def __init__(self):
        self.running_e1, self.running_e2 = 0, 0
        self.count = 0
    
    def __call__(self, ds):
        if self.count < 20:
            x = ds[0]
            e1, e2 = eigens(patches(torch.from_numpy(x)))
            self.running_e1 += e1
            self.running_e2 += e2
            self.count += 1
            ParameterManager.save_buffer('e1', self.running_e1 / self.count)
            ParameterManager.save_buffer('e2', self.running_e2 / self.count)
            ParameterManager.save_buffer('preloop_count', self.count)
        return ds

    
class _Preloop(nn.Module):
    def __init__(self):
        super().__init__()
        self.train_ds = None
        self.val_ds = None
        self.collector = collect_statistics()
        self.phase = 0
    
    def set_phase(self, phase):
        self.phase = phase
        
    def clear_source(self, is_train):
        try:
            del self.train_ds
        except:
            pass
        
        try:
            del self.val_ds
        except:
            pass
            
        self.train_ds = None
        self.val_ds = None
        self.collector = collect_statistics()
    
    def forward(self, ds):
        if self.train_ds is None and self.phase == 0:
            self.train_ds = MapData(ds, self.collector)
        elif self.val_ds is None and self.phase == 1:
            self.val_ds = ds
            
        if self.phase == 0:
            return self.train_ds
        elif self.phase == 1:
            return self.val_ds
        else:
            return -1 # Custom predict phase
