from ..template import template


class Normalizer(template.TorchPreprocessing):

    def __init__(self, include_y=False):
        super().__init__()
        self.include_y = include_y

    def _scale_fn(self, var):
        var -= var.mean()
        var /= var.std()
        return var

    def forward_batch(self, x, y):
        x = [self._scale_fn(i) for i in x]
        if self.include_y:
            y = [self._scale_fn(i) for i in y]
        return x, y
