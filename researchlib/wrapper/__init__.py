from .time_distributed import *

from .multiscale import _MultiscaleOutput, _MultiscaleInput
from .auxiliary import _Auxiliary
from .equalize_lr import _EqualizeLr
from .identical import _Identical
from .squeeze import _Squeeze
from .unsqueeze import _Unsqueeze
from .regularize import _Regularize
from .time_distributed import _TimeDistributed

class wrapper(object):
    EqualizeLr=_EqualizeLr
    Auxiliary=_Auxiliary
    Identical=_Identical
    Unsqueeze=_Unsqueeze
    Squeeze=_Squeeze
    Regularize=_Regularize
    TimeDistributed=_TimeDistributed
    MultiscaleOutput=_MultiscaleOutput
    MultiscaleInput=_MultiscaleInput