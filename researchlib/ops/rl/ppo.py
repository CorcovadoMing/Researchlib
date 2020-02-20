from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import _discount_returns


class _PPO(nn.Module):
    def __init__(self, agent, epsilon=0.1, state_node='state', policy_node='policy'):
        super().__init__()
        self.epsilon = epsilon
        self.eps = np.finfo(np.float32).eps.item()
        self.agent = agent
        self.state_node = state_node
        self.policy_node = policy_node
        self.device = None
        self.fixed_log_prob = []
    
    def _fixed_log_prob(self, trajection):
        with torch.no_grad():
            if self.device is None:
                self.device = next(self.agent.parameters()).device
            result = self.agent({self.state_node: torch.stack(trajection['state'], 0).to(self.device)})
            action = torch.LongTensor(trajection['action']).to(self.device)
            logp = result[self.policy_node].log_prob(action).view(-1)
        return logp
    
    def _process_single(self, trajection, fixed_logp):
        if self.device is None:
            self.device = next(self.agent.parameters()).device
        result = self.agent({self.state_node: torch.stack(trajection['state'], 0).to(self.device)})
        action = torch.LongTensor(trajection['action']).to(self.device)
        logp = result[self.policy_node].log_prob(action).view(-1)
        with torch.no_grad():
            returns = torch.from_numpy(_discount_returns(trajection['reward'])).to(self.device)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
            weights = returns.clone()
            weigths = weights - result['value']
        ratio = torch.exp(logp - fixed_logp)
        surrogate1 = ratio * weigths
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * weigths
        surrogate = -torch.min(surrogate1, surrogate2)
        return surrogate, result['value'], returns
    
    def forward(self, eps_trajection, inner_loop=0):
        self.agent.train()
        
        # Get the fixed referenced policy
        if inner_loop == 0:
            self.fixed_log_prob = []
            for i in eps_trajection:
                self.fixed_log_prob.append(self._fixed_log_prob(i))
        
        loss = 0
        for trajection, fixed_logp in zip(eps_trajection, self.fixed_log_prob):
            surrogate, values, returns = self._process_single(trajection, fixed_logp)
            loss += surrogate.sum() + F.smooth_l1_loss(values, returns.view_as(values), reduction='sum')
        return loss / len(eps_trajection)

