from .prefetch import BackgroundGenerator
from ...utils import inifinity_loop
from torch import nn


class _Source(nn.Module):
    def __init__(self, train_source, val_source = None, **kwargs):
        super().__init__()
        self.train_source = train_source
        self.train_source_generator = None
        self.val_source = val_source
        self.val_source_generator = None
        self.kwargs = kwargs
        
    def clear_source(self, is_train):
        del self.train_source_generator
        self.train_source_generator = None
        del self.val_source_generator
        self.val_source_generator = None
        
    def prepare_generator(self, epochs):
        if self.train_source_generator is None and self.train_source is not None:
            self.train_source_generator = self.train_source.get_generator(epochs=epochs, **self.kwargs)
        if self.val_source_generator is None and self.val_source is not None:
            self.val_source_generator = self.val_source.get_generator(epochs=epochs)
    
    def forward(self, x):
        if x == 0:
            return self.train_source_generator
        elif x == 1:
            return self.val_source_generator