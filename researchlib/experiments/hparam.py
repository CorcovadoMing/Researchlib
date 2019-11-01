import numpy as np

def Sweep(min, max, step):
    return np.linspace(min, max, int(((max-min)/step)+1))