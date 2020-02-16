from .simulator import _Simulator
from .vanilla_policy_gradient import _VanillaPG

class RL(object):
    Simulator = _Simulator
    VanillaPG = _VanillaPG