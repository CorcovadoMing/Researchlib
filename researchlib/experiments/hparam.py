import numpy as np

def Sweep(min, max, step):
    return np.linspace(min, max, int(((max-min)/step)+1))

def Predef(list_of_param):
    return np.array(list_of_param)