import torch 


class NumpyPreprocessing(object):
    def __init__(self):
        pass
    
    def _forward(self, x, y):
        x, y = [i.numpy() for i in x], [i.numpy() for i in y]
        x, y = self.forward(x, y)
        try:
            x, y = [torch.from_numpy(i) for i in x], [torch.from_numpy(i) for i in y]
        except:
            x, y = [torch.from_numpy(i.copy()) for i in x], [torch.from_numpy(i.copy()) for i in y]
        return x, y
    
    def forward(self, x, y):
        raise('Not implemented')

        
class TorchPreprocessing(object):
    def __init__(self):
        pass
    
    def _forward(self, x, y):
        x, y = self.forward(x, y)
        return x, y
    
    def forward(self, x, y):
        raise('Not implemented')
        
        
class template(object):
    NumpyPreprocessing = NumpyPreprocessing
    TorchPreprocessing = TorchPreprocessing