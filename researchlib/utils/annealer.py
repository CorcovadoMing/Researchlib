import numpy as np


def _Cosine(tcur, srange, tmax):
        start, end = srange
        if tcur <= tmax:
            return end + (1/2) * (start - end) * (1 + np.cos(tcur/tmax * np.pi))
        else:
            return end

        
class Annealer:
    tracker = {}
    
    Cosine = _Cosine
    
    @classmethod
    def set_trace(cls, name, max_step, srange=[0, 1], anneal_when='iteration', anneal_fn=lambda x: x):
        cls.tracker[name] = {'value': srange[0], 'srange': srange, 'anneal_fn': anneal_fn, 'anneal_when': anneal_when, 'cur_step': 0, 'max_step': max_step}
        return cls # for checking
    
    @classmethod
    def get_trace(cls, name):
        return cls.tracker[name]['value']
    
    @classmethod
    def _iteration_step(cls):
        for key in cls.tracker:
            if cls.tracker[key]['anneal_when'] == 'iteration':
                cls.tracker[key]['cur_step'] += 1
                cls.tracker[key]['value'] = cls.tracker[key]['anneal_fn'](cls.tracker[key]['cur_step'], cls.tracker[key]['srange'], cls.tracker[key]['max_step'])

    @classmethod
    def _epoch_step(cls):
        for key in cls.tracker:
            if cls.tracker[key]['anneal_when'] == 'epoch':
                cls.tracker[key]['cur_step'] += 1
                cls.tracker[key]['value'] = cls.tracker[key]['anneal_fn'](cls.tracker[key]['cur_step'], cls.tracker[key]['srange'], cls.tracker[key]['max_step'])

