import numpy as np
from tensorpack.dataflow import *
from tensorpack import imgaug


class _CIFAR10:
    def __init__(self, is_train):
        self.is_train = is_train
        phase = 'train' if is_train else 'test'
        self.ds = dataset.Cifar10(phase, shuffle=is_train)
        
        if is_train:
            # Record the statistic for normalizer 
            l = np.array([i[0] for i in self.ds.data])
            _CIFAR10.mean, _CIFAR10.std = np.mean(l, axis=tuple(range(l.ndim-1))), np.std(l, axis=tuple(range(l.ndim-1)))
            del l
        
        self.normalizer = []
        self.augmentor = []
        
        
    def _set_normalizer(self):
        self.normalizer = [imgaug.MapImage(lambda x: (x - _CIFAR10.mean)/_CIFAR10.std)]
    
    
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
        self.augmentor.append(imgaug.RandomOrderAug(_aug))
        
        
    def get_generator(self, batch_size=512):
        if self.is_train:
            ops = self.augmentor + self.normalizer
        else:
            ops = self.normalizer

        ds = AugmentImageComponent(self.ds, ops)

        def mapf(dp):
            x, y = dp
            return np.moveaxis(x, -1, 0).astype(np.float32), np.array(y).astype(np.int64)

        ds = MapData(ds, mapf)
        ds = BatchData(ds, batch_size, remainder=True)
        if self.is_train:
            ds = MultiProcessPrefetchData(ds, 8, 8)
        ds = PrintData(ds)
        ds.reset_state()
        return ds