from .simulator import _Simulator
from .reinforce import _REINFORCE
from .actor_critic import _ActorCritic
from .ppo import _PPO


class RL(object):
    Simulator = _Simulator
    REINFORCE = _REINFORCE
    ActorCritic = _ActorCritic
    PPO = _PPO