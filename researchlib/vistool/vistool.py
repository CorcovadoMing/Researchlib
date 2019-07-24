from .data import _Image
from .model import _Model
from .match import _Match


class _Vistool:
    def __init__(self):
        self.image = _Image()
        self.model = _Model()
        self.match = _Match()
