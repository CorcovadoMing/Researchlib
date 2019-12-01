from .prefetch import BackgroundGenerator
from ...utils import inifinity_loop


class _Source:
    def __init__(self, train_source, val_source = None, **kwargs):
        self.train_source = train_source
        self.train_source_generator = None
        self.val_source = val_source
        self.val_source_generator = None
        self.kwargs = kwargs
        
    def prepare_generator(self, batch_size, epochs):
        if self.train_source_generator is None and self.train_source is not None:
            self.train_source_generator = self.train_source.get_generator(batch_size=batch_size, epochs=epochs, **self.kwargs)
        if self.val_source_generator is None and self.val_source is not None:
            self.val_source_generator = self.val_source.get_generator(batch_size=batch_size, epochs=epochs)
    
    def __call__(self, x):
        if x == 0:
            return self.train_source_generator
        elif x == 1:
            return self.val_source_generator