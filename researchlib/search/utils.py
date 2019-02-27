import math
import random

def best(fo, fc, opt):
    return True

def identity(opt, epoch):
    return opt

def boltzmann(fo, fc, opt):
    e = min(abs(fo-fc) / opt['t_cur'], 500)
    p = math.exp(e)
    return p > random.random()

def annealing(opt, epoch):
    opt['t_cur'] *= opt['anneal_rate']
    opt['t_cur'] = max(opt['t_cur'], 1e-5)
    return opt
