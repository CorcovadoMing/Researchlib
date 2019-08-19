from .data import _Image
from .model import _Model
from .match import _Match
from .interpreter import _Interpreter


class _Vistool:

    def __init__(self):
        self.image = _Image()
        self.model = _Model()
        self.match = _Match()
        self.interpreter = _Interpreter()
