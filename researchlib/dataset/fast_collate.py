import torch
import numpy as np


def fast_collate(batch):
    '''
        Adapted from
        https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L77
    '''
    imgs = [img[0] for img in batch]
    t = [target[1] for target in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.float32)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.float32)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets