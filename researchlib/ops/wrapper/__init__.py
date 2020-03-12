from .equalize_lr import _EqualizeLr
from .time_distributed import _TimeDistributed
from .random_network_distillation import _RandomNetworkDistillation
from .intrinsic_curiosity import _IntrinsicCuriosity


class Wrapper(object):
    EqualizeLr = _EqualizeLr
    TimeDistributed = _TimeDistributed
    RandomNetworkDistillation = _RandomNetworkDistillation
    IntrinsicCuriosity = _IntrinsicCuriosity