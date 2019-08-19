from .callback import Callback
from ..utils import *
import math
import numpy as np


class AdaptiveFocalUpdate(Callback):

    def __init__(self):
        super().__init__()
        self.total_map = None
        self.correct_map = None

    def on_iteration_end(self, **kwargs):
        if kwargs['model'].training == False:
            output, target = kwargs['auxout'][-1], kwargs['target'][-1]
            if self.total_map == None:
                self.total_map = [0 for _ in range(output.size(1))]
            if self.correct_map == None:
                self.correct_map = [0 for _ in range(output.size(1))]

            output = output.argmax(1)
            for i in range(len(output)):
                if output[i] == target[i].long():
                    self.correct_map[target[i].long()] += 1
                self.total_map[target[i].long()] += 1

        return kwargs

    def on_validation_end(self, **kwargs):
        alpha = [(float(self.correct_map[i]) / self.total_map[i])
                 for i in range(len(self.total_map))]
        print(alpha)
        print(np.array(alpha).std())
        alpha = [(1 - i) for i in alpha]
        alpha = torch.from_numpy(np.array(alpha)).float().cuda()
        self.total_map = None
        self.correct_map = None
        for i in range(len(kwargs['loss_fn'])):
            try:
                kwargs['loss_fn'][i].update_alpha(alpha)
            except:
                pass
        return kwargs
