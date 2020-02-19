from torch import nn
import torch
import numpy as np


def _reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class _REINFORCE(nn.Module):
    def __init__(self, agent, state_node='state', policy_node='policy'):
        super().__init__()
        self.eps = np.finfo(np.float32).eps.item()
        self.agent = agent
        self.state_node = state_node
        self.policy_node = policy_node
        self.device = None
        
    def _process_single(self, trajection):
        if self.device is None:
            self.device = next(self.agent.parameters()).device
        result = self.agent({self.state_node: torch.stack(trajection['state'], 0).to(self.device)})
        logp = result[self.policy_node].log_prob(torch.LongTensor(trajection['action']).to(self.device))
        returns = torch.FloatTensor(list(_reward_to_go(trajection['reward']))).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        weights = returns.clone()
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
        
        logp = torch.cat(logp, 0).to(self.device)
        weights = torch.cat(weights, 0).to(self.device)
        return -(logp * weights).mean()