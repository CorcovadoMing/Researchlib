import numpy as np


def _Cosine(tcur, srange, tmax):
    start, end = srange
    if tcur <= tmax:
        return end + (1 / 2) * (start - end) * (1 + np.cos(tcur / tmax * np.pi))
    else:
        return end


def _CosineTwoWay(tcur, srange, tmax, split_ratio = 0.333):
    split_epoch = int(tmax * split_ratio)
    if tcur <= split_epoch:
        return _Cosine(tcur, srange, split_epoch)
    else:
        return _Cosine(tcur - split_epoch, srange[::-1], tmax - split_epoch)


def _Linear(tcur, srange, tmax):
    start, end = srange
    if tcur <= tmax:
        return start + (end - start) * (tcur / tmax)
    else:
        return end


def _Fixed(tcur, srange, tmax):
    start, end = srange
    return max(start, end)


class Annealer:
    tracker = {}

    Cosine = _Cosine
    CosineTwoWay = _CosineTwoWay
    Linear = _Linear
    Fixed = _Fixed

    @classmethod
    def set_trace(
        cls, name, max_step, srange = [0, 1], anneal_when = 'iteration', anneal_fn = lambda x: x
    ):
        cls.tracker[name] = {
            'value': srange[0],
            'srange': srange,
            'anneal_fn': anneal_fn,
            'anneal_when': anneal_when,
            'cur_step': 0,
            'max_step': max_step
        }
        return cls  # for checking

    @classmethod
    def get_trace(cls, name):
        return cls.tracker[name]['value']

    @classmethod
    def get_srange(cls, name):
        return cls.tracker[name]['srange']

    @classmethod
    def update_attr(cls, name, key, value):
        cls.tracker[name][key] = value

    @classmethod
    def _iteration_step(cls, key = None):
        if key is not None:
            if cls.tracker[key]['anneal_when'] == 'iteration':
                cls.tracker[key]['cur_step'] += 1
                cls.tracker[key]['value'] = cls.tracker[key]['anneal_fn'](
                    cls.tracker[key]['cur_step'], cls.tracker[key]['srange'],
                    cls.tracker[key]['max_step']
                )
        else:
            for _key in cls.tracker:
                if cls.tracker[_key]['anneal_when'] == 'iteration':
                    cls.tracker[_key]['cur_step'] += 1
                    cls.tracker[_key]['value'] = cls.tracker[_key]['anneal_fn'](
                        cls.tracker[_key]['cur_step'], cls.tracker[_key]['srange'],
                        cls.tracker[_key]['max_step']
                    )

    @classmethod
    def _epoch_step(cls, key = None):
        if key is not None:
            if cls.tracker[key]['anneal_when'] == 'epoch':
                cls.tracker[key]['cur_step'] += 1
                cls.tracker[key]['value'] = cls.tracker[key]['anneal_fn'](
                    cls.tracker[key]['cur_step'], cls.tracker[key]['srange'],
                    cls.tracker[key]['max_step']
                )
        else:
            for _key in cls.tracker:
                if cls.tracker[_key]['anneal_when'] == 'epoch':
                    cls.tracker[_key]['cur_step'] += 1
                    cls.tracker[_key]['value'] = cls.tracker[_key]['anneal_fn'](
                        cls.tracker[_key]['cur_step'], cls.tracker[_key]['srange'],
                        cls.tracker[_key]['max_step']
                    )
