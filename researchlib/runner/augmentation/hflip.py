from ..template import template


class HFlip(template.NumpyAugmentation):
    def __init__(self, include_y=False):
        super().__init__()
        self.include_y = include_y
    
    def forward_single(self, x, y, mag):
        x = [ i[:, :, ::-1] for i in x]
        if self.include_y:
            y = [ i[..., ::-1] for i in y]
        return x, y