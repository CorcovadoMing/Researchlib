from .patch import _Patch
from .mask_and_replace import _MaskAndReplace
from .loss_mask import _LossMask


class N2V(object):
    Patch = _Patch
    MaskAndReplace = _MaskAndReplace
    LossMask = _LossMask
    