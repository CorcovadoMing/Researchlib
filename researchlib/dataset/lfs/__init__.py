from .generative import Generative
from .classification import Classification


class _LFS(object):
    Generative = Generative
    Classification = Classification