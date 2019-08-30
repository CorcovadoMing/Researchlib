class Annealer:
    tracker = {}
    
    @classmethod
    def set_trace(cls, name, value, anneal_when='iteration', anneal_fn=lambda x: x):
        cls.tracker[name] = {'value': value, 'anneal_fn': anneal_fn, 'anneal_when': anneal_when}
    
    @classmethod
    def get_trace(cls, name):
        return cls.tracker[name]['value']
    
    @classmethod
    def _iteration_step(cls):
        for key in cls.tracker:
            if cls.tracker[key]['anneal_when'] == 'iteration':
                cur_value = cls.tracker[key]['value']
                cls.tracker[key]['value'] = cls.tracker[key]['anneal_fn'](cur_value)
    
    @classmethod
    def _epoch_step(cls):
        for key in cls.tracker:
            if cls.tracker[key]['anneal_when'] == 'epoch':
                cur_value = cls.tracker[key]['value']
                cls.tracker[key]['value'] = cls.tracker[key]['anneal_fn'](cur_value)
    