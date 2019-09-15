from ..template import template
import torchvision


class RandomErasing(template.TorchAugmentation):

    def __init__(self, prob=None, mag=None, include_y=False):
        super().__init__()
        self.include_y = include_y
        self.prob = prob
        self.mag = mag
        self.fn = torchvision.transforms.RandomErasing(
            p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')

    def forward_single(self, x, y, mag):
        x = [self.fn(i) for i in x]
        if self.include_y:
            y = [self.fn(i) for i in y]
        return x, y
