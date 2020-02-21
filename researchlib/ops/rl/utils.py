import numpy as np


def _discount_returns(rewards, discount=0.99):
    n = len(rewards)
    returns = np.zeros(n).astype(np.float32)
    for i in reversed(range(n)):
        returns[i] = rewards[i] + discount * (returns[i+1] if i+1 < n else 0)
    return returns