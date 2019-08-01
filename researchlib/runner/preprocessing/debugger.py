from .template import template


class Debugger(template.TorchPreprocessing):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        for i in x:
            print('X Min:', i.min(), 'X Max:', i.max(), 'X Std:', i.float().std(), 'X Shape:', i.shape)
        for i in y:
            print('Y Min:', i.min(), 'Y Max:', i.max(), 'Y Std:', i.float().std(), 'Y Shape:', i.shape)
        return x, y
            