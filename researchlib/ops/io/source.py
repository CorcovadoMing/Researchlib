from .prefetch import BackgroundGenerator
from ...utils import inifinity_loop
from torch import nn


class _Source(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.source_generator = None
    
    def prepare_source(self, batch_size, epochs, fp16):
        del self.source_generator
        self.source_generator = BackgroundGenerator(inifinity_loop(self.source.get_generator(batch_size, epochs=epochs)), fp16=fp16)
        
    def forward(self, x):
        return next(iter(self.source_generator))