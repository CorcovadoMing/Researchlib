from .callback import Callback

class Binarized(Callback):
    def __init__(self):
        super().__init__()
        
    def on_update_begin(self, **kwargs):
        for p in list(kwargs['model'].parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
    
    def on_update_end(self, **kwargs):
        for p in list(kwargs['model'].parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))