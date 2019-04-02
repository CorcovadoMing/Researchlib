from .callback import Callback
from ..utils import *
import math

class AdaptiveFocalUpdate(Callback):
    def __init__(self):
        super().__init__()

    def on_iteration_end(self, **kwargs):
        if kwargs['model'].training == False:
            pass
        return kwargs

    def on_validation_end(self, **kwargs):
        print(kwargs['metrics'])
        return kwargs