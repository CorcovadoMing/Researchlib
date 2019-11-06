from torch import nn


class _OneHot(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes
    
    def forward(self, x):
        return nn.functional.one_hot(x.long(), self.classes)