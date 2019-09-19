from ..template import template


class MinMaxScaler(template.TorchPreprocessing):
    def __init__(self, target_range = [-1, 1], include_y = False):
        super().__init__()
        self.target_range = target_range
        self.include_y = include_y

    def _scale_fn(self, var):
        # self-normalize to [0, 1]
        var -= var.min()
        var /= var.max()

        # mapping to target distribution
        tmin, tmax = self.target_range[0], self.target_range[1]
        trange = tmax - tmin
        var *= trange
        var += tmin

        return var

    def forward_single(self, x, y):
        x = [self._scale_fn(i) for i in x]
        if self.include_y:
            y = [self._scale_fn(i) for i in y]
        return x, y
