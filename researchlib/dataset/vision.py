import numpy as np
from tensorpack.dataflow import *
from tensorpack import imgaug


class _VISION_GENERAL_LOADER:
    def __init__(self, is_train, ds):
        self.is_train = is_train
        self.ds = ds
        
        if is_train:
            # Record the statistic for normalizer 
            l = np.array([i[0] for i in self.ds.data])
            _VISION_GENERAL_LOADER.mean, _VISION_GENERAL_LOADER.std = np.mean(l, axis=tuple(range(l.ndim-1))), np.std(l, axis=tuple(range(l.ndim-1)))
            del l
        
        self.normalizer = []
        self.augmentor = []
        
        
    def _set_normalizer(self, local=False):
        if not local:            
            self.normalizer = [imgaug.MapImage(lambda x: (x - _VISION_GENERAL_LOADER.mean)/_VISION_GENERAL_LOADER.std)]
        else:
            self.normalizer = [imgaug.MeanVarianceNormalize()]
    
    
    def _set_augmentor(self, augmentor):
        mapping = {
            'hflip': imgaug.Flip(horiz=True),
            'crop': imgaug.RandomApplyAug(
                        imgaug.AugmentorList([
                            imgaug.CenterPaste((40, 40)),
                            imgaug.RandomCrop((32, 32)),
                        ]),
                    0.5),
            'cutout': imgaug.RandomApplyAug(imgaug.RandomCutout(8, 8), 0.5)
        }
        
        _aug = []
        for i in augmentor:
            _aug.append(mapping[i])
        self.augmentor = [imgaug.RandomOrderAug(_aug)]
        
        
    def get_generator(self, batch_size=512, **kwargs):
        
        def batch_mapf(dp):
            x = dp[0].copy()
            y = dp[1]
            
            y_type = str(y.dtype)
            if 'int' in y_type:
                y_type = np.int64
            else:
                y_type = np.float32
            
            if self.is_train:
                for i in range(len(x)):
                    for op in self.augmentor:
                        x[i] = op.augment(x[i])
            
            for op in self.normalizer:
                x = op.augment(x)
            
            return np.moveaxis(x, -1, 1).astype(np.float32), np.array(y).astype(y_type)

        ds = BatchData(self.ds, batch_size, remainder=True)
        ds = MultiProcessMapDataZMQ(ds, 4, batch_mapf)
        if self.is_train:
            ds = MultiProcessPrefetchData(ds, 2048//batch_size, 4)
        ds = PrintData(ds)
        ds.reset_state()
        return ds
    
    
    
def _CIFAR10(is_train=True):
    phase = 'train' if is_train else 'test'
    return _VISION_GENERAL_LOADER(is_train, dataset.Cifar10(phase, shuffle=is_train))


def _NumpyDataset(x, y, is_train=True):
    _inner_gen = DataFromList(list(zip(x, y)), shuffle=is_train)
    _inner_gen.data = x
    return _VISION_GENERAL_LOADER(is_train, _inner_gen)