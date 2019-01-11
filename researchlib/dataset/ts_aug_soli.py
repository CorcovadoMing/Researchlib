import random
import copy
import numpy as np

#-------------------------------------
# Augmentation for Soli Radar (bs, ts, 2, 32, 32)
#-------------------------------------

def shift(x_, y_, s_min=-10, s_max=10):
    '''
        Horizontal Shift (Doppler Range Shift)
    '''
    ratio = np.random.randint(s_min, s_max, 1)[0]
    if ratio == 0:
        return x_, y_
    else:
        x = copy.deepcopy(x_)
        if ratio > 0:
            x[:, :, :, :-ratio, :] = x[:, :, :, ratio:, :]
            x[:, :, :, -ratio:, :] = 0
        else:
            x[:, :, :, -ratio:, :] = x[:, :, :, :ratio, :] 
            x[:, :, :, :-ratio, :] = 0
        return x, y_

def noise(x_, y_, n_min=1, n_max=15, p=0.25):
    '''
        Uniform Random Noise
    '''
    x = copy.deepcopy(x_)
    n = np.random.randint(n_min, n_max, x.shape)
    m = np.random.random(x.shape)
    m = m < p
    n = n * m
    return x+n, y_

def gain(x_, y_, s_min=0.75, s_max=1.25):
    '''
        Uniform Random Gain (Texture of Reflector)
    '''
    x = copy.deepcopy(x_)
    ratio = np.random.uniform(s_min, s_max, 1)[0]
    return x*ratio, y_

def dropf(x_, y_, p=0.25):
    '''
        Drop Frames
    '''
    x = copy.deepcopy(x_)
    n = np.random.random(x.shape[1])
    m = n < p
    x[:, m, :, :, :] = 0
    return x, y_ # y need to be manipulated?

def ssvi(x_, y_, s_min=2, s_max=4):
    '''
        Sample Speed Variants (Increase)
    '''
    s = np.random.randint(s_min, s_max, 1)[0]
    x = copy.deepcopy(x_)
    y = copy.deepcopy(y_)
    m = list(range(0, x.shape[1], s))
    return np.delete(x, m, 1), np.delete(y, m, 1)

def ssvd(x_, y_, s_min=1, s_max=2):
    '''
        Sample Speed Variants (Decrease)
    '''
    s = np.random.randint(s_min, s_max, 1)[0]
    d = int(x_.shape[1]/s)
    x_sh = list(x_.shape)
    x_sh[1] += d
    y_sh = list(y_.shape)
    y_sh[1] += d
    x = np.zeros(x_sh)
    y = np.zeros(y_sh)
    m = list(range(0, x_.shape[1], s))
    counter = 0
    for i in range(x_.shape[1]):
        if i in m:
            if i == 0:
                x[:, i+counter] = x_[:, i]
                y[:, i+counter] = y_[:, i]
            else:
                x[:, i+counter] = (x[:, i+counter-1]+x_[:, i])/2.
                y[:, i+counter] = y_[:, i]
            counter += 1
        x[:, i+counter] = x_[:, i]
        y[:, i+counter] = y_[:, i]
    return x, y