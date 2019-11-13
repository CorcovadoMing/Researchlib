from .data import _Image
from .model import _Model
from .match import _Match
from .interpreter import _Interpreter
from .summary import _summary

class VisKit(object):
    image = _Image()
    model = _Model()
    match = _Match()
    interpreter = _Interpreter()
    summary = _summary