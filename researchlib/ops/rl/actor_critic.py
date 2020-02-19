from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import _discount_returns


class _ActorCritic(nn.Module):
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
        returns = torch.from_numpy(_discount_returns(trajection['reward'])).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        weights = returns.clone()
        weigths = weights - result['value']
        return logp, weights, result['value'], returns
    
    def forward(self, eps_trajection):
        self.agent.train()

        logp = []
        weights = []
        values = []
        returns = []
        
        for i in eps_trajection:
            _logp, _weights, _values, _returns = self._process_single(i)
            logp.append(_logp)
            weights.append(_weights)
            values.append(_values)
            returns.append(_returns)
            
        
        logp = torch.cat(logp, 0).to(self.device)
        weights = torch.cat(weights, 0).to(self.device)
        values = torch.cat(values, 0).to(self.device)
        returns = torch.cat(returns, 0).to(self.device)
        return -(logp * weights).mean() + F.smooth_l1_loss(values, returns.view_as(values))
