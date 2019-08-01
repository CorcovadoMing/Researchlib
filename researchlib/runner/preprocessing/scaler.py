from .template import template


class Scaler(template.TorchPreprocessing):
    def __init__(self, target_range = [-1, 1], include_y=False):
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
    
    def forward(self, x, y):
        for inputs in range(len(x)):
            for batch in range(len(x[inputs])):
                x[inputs][batch] = self._scale_fn(x[inputs][batch])
        if self.include_y:
            for inputs in range(len(y)):
                for batch in range(len(y[inputs])):
                    y[inputs][batch] = self._scale_fn(y[inputs][batch])
        return x, y
            