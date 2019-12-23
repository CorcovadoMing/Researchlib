from .generative import Generative
from .classification import Classification
from .medical import Medical
from .restoration import Restoration


class _LFS(object):
    Generative = Generative
    Classification = Classification
    Medical = Medical
    Restoration = Restoration