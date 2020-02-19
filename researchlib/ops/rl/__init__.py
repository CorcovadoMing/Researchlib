from .simulator import _Simulator
from .reinforce import _REINFORCE
from .actor_critic import _ActorCritic


class RL(object):
    Simulator = _Simulator
    REINFORCE = _REINFORCE
    ActorCritic = _ActorCritic