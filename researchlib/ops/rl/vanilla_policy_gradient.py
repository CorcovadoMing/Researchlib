from torch import nn
import torch
import numpy as np


def _reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class _VanillaPG(nn.Module):
    def __init__(self, agent, state_node='state', policy_node='policy'):
        super().__init__()
        self.agent = agent
        self.state_node = state_node
        self.policy_node = policy_node
        
    def _process_single(self, trajection):
        result = self.agent({self.state_node: torch.stack(trajection['state'], 0)})
        logp = result[self.policy_node].log_prob(torch.LongTensor(trajection['action']))
        weights = torch.FloatTensor(list(_reward_to_go(trajection['reward'])))
        return logp, weights
    
    def forward(self, eps_trajection):
        self.agent.train()

        logp = []
        weights = []
        returns = 0
        
        for i in eps_trajection:
            _logp, _weights = self._process_single(i)
            logp.append(_logp)
            weights.append(_weights)
            returns += len(_logp)
    
        print(returns / len(eps_trajection))
        
        logp = torch.cat(logp, 0)
        weights = torch.cat(weights, 0)
        return -(logp * weights).mean()