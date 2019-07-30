from .parser import _Parser
from .loader import _Loader


class _Datatool:
    def __init__(self):
        self.parser = _Parser()
        self.loader = _Loader()
