from .callback import Callback

class LRRangeTest(Callback):
    def __init__(self, iterations, max_lr=10, min_lr=1e-5):
        super(LRRangeTest, self).__init__()
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step = (self.max_lr / self.min_lr) ** (1 / float(iterations))

    def on_iteration_begin(self, **kwargs):
        cur_lr = self.min_lr * (self.step ** kwargs['batch_idx'])
        for g in kwargs['optimizer'].param_groups:
            g['lr'] = cur_lr