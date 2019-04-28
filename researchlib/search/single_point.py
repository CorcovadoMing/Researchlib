
import torch
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

# -----------------------------------------------------------


def sp_meta_heuristic_(x, objective, epoch, mode, accept, opt, opt_tune):
    records = []
    fitness = objective(x)
    x_record = x
    for _ in range(epoch):
        x_ = x_record.clone()
        x_ += torch.randn(x_.size()).cuda()
        fitness_ = objective(x_)
        if mode(fitness, fitness_) == fitness_:
            if accept(fitness, fitness_, opt):
                x_record = x_
                fitness = fitness_
        records.append(fitness)
        opt = opt_tune(opt, epoch)
    return x_record, fitness, records


# Algorithm Interface
# -----------------------------------------------------------


def _IterativeImprovement(x, objective, epoch, mode=min):
    return sp_meta_heuristic_(x, objective, epoch, mode, accept=best, opt={}, opt_tune=identity)

def _SimulatedAnnealing(x, objective, epoch, mode=min, t_max=100, t_rate=0.95):
    return sp_meta_heuristic_(x, objective, epoch, mode, accept=boltzmann, opt={'t_cur': t_max, 'anneal_rate': t_rate}, opt_tune=annealing)