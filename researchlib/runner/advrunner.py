from .runner import Runner
from .advtrain import *

class AdvRunner(Runner):
    def __init__(self, model=None, train_loader=None, test_loader=None, optimizer=None, loss_fn=None):
        super().__init__(model, train_loader, test_loader, optimizer, loss_fn)
        self.trainer = advtrain
        