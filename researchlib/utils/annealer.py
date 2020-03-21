import numpy as np


def _Cosine(tcur, srange, tmax):
    start, end = srange
    if tcur <= tmax:
        return end + (1 / 2) * (start - end) * (1 + np.cos(tcur / tmax * np.pi))
    else:
        return end

def _Linear(tcur, srange, tmax):
    start, end = srange
    if tcur <= tmax:
        return start + (end - start) * (tcur / tmax)
    else:
        return end

def _Poly2(tcur, srange, tmax):
    start, end = srange
    if tcur <= tmax:
        return start + (end - start) * (tcur / tmax) ** 2
    else:
        return end
    
def _Fixed(tcur, srange, tmax):
    start, end = srange
    return max(start, end)


class Annealer:
    
    Cosine = _Cosine
    Linear = _Linear
    Fixed = _Fixed
    Poly2 = _Poly2

    def __init__(self):
        self.tracker = {}

    def reset(self):
        cls.tracker = {}
    
    def set_trace(self, 
                    name, 
                    max_step, 
                    srange = [0, 1], 
                    anneal_when = 'iteration', 
                    anneal_fn = lambda x: x,
                    force = False):
        if force or name not in self.tracker:
            self.tracker[name] = {
                'value': srange[0],
                'srange': srange,
                'anneal_fn': anneal_fn,
                'anneal_when': anneal_when,
                'cur_step': 0,
                'max_step': max_step
            }
            return True
        else:
            return False

    def get_trace(self, name):
        return self.tracker[name]['value']

    def _post_config(self, epoch, iterations):
        for key in self.tracker:
            if self.tracker[key]['max_step'] is None:
                if self.tracker[key]['anneal_when'] == 'epoch':
                    self.tracker[key]['max_step'] = epoch
                if self.tracker[key]['anneal_when'] == 'iteration':
                    self.tracker[key]['max_step'] = epoch * iterations
    
    def get_srange(self, name):
        return self.tracker[name]['srange']

    def update_attr(self, name, key, value):
        self.tracker[name][key] = value

    def _iteration_step(self, key = None):
        if key is not None:
            if self.tracker[key]['anneal_when'] == 'iteration':
                self.tracker[key]['cur_step'] += 1
                self.tracker[key]['value'] = self.tracker[key]['anneal_fn'](
                    self.tracker[key]['cur_step'], self.tracker[key]['srange'],
                    self.tracker[key]['max_step']
                )
        else:
            for _key in self.tracker:
                if self.tracker[_key]['anneal_when'] == 'iteration':
                    self.tracker[_key]['cur_step'] += 1
                    self.tracker[_key]['value'] = self.tracker[_key]['anneal_fn'](
                        self.tracker[_key]['cur_step'], self.tracker[_key]['srange'],
                        self.tracker[_key]['max_step']
                    )

    def _epoch_step(self, key = None):
        if key is not None:
            if self.tracker[key]['anneal_when'] == 'epoch':
                self.tracker[key]['cur_step'] += 1
                self.tracker[key]['value'] = self.tracker[key]['anneal_fn'](
                    self.tracker[key]['cur_step'], self.tracker[key]['srange'],
                    self.tracker[key]['max_step']
                )
        else:
            for _key in self.tracker:
                if self.tracker[_key]['anneal_when'] == 'epoch':
                    self.tracker[_key]['cur_step'] += 1
                    self.tracker[_key]['value'] = self.tracker[_key]['anneal_fn'](
                        self.tracker[_key]['cur_step'], self.tracker[_key]['srange'],
                        self.tracker[_key]['max_step']
                    )
