from torch import nn
import torch
import numpy as np
from .utils import _discount_returns


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
        action = torch.LongTensor(trajection['action']).to(self.device)
        logp = result[self.policy_node].log_prob(action).view(-1)
        with torch.no_grad():
            logq = torch.stack([i.log_prob(j) for i, j in zip(trajection['policy'], action)], 0).view(-1)
            returns = torch.from_numpy(_discount_returns(trajection['reward'])).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
            weights = returns.clone()
            weights = weights * (logp.exp() / logq.exp()).clamp_(0, 2)
        return logp, weights
    
    def forward(self, eps_trajection):
        self.agent.train()
        loss = 0
        for i in eps_trajection:
            logp, weights = self._process_single(i)
            loss += -(logp * weights).sum()
        return loss / len(eps_trajection)