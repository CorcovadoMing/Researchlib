from ..template import template
from torchvision import transforms


class Normalizer(template.TorchPreprocessing):

    def __init__(self, static_auto=True, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.static_auto = static_auto

    def _scale_fn(self, var, static_auto):
        if static_auto:
            for i in range(var.size(0)):
                var[i] = self.static_normalizer(var[i])
        else:
            var -= var.mean()
            var /= var.std()
        return var

    def forward_batch(self, x, y):
        x = [self._scale_fn(i, static_auto=self.static_auto) for i in x]
        if self.include_y:
            y = [self._scale_fn(i, static_auto=self.static_auto) for i in y]
        return x, y
