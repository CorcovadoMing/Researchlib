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
            returns = torch.from_numpy(_discount_returns(trajection['reward'])).to(self.device).view(-1)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
            intrinsic = torch.zeros_like(returns) if 'intrinsic' not in trajection else trajection['intrinsic']
            intrinsic = torch.from_numpy(_discount_returns(trajection['reward'], 0.9)).to(self.device).view(-1)
        value_rollout = result['value'].view(-1)
        intrinsic_rollout = torch.zeros_like(value_rollout) if 'intrinsic' not in trajection \
                                                            else result['intrinsic'].view(-1)
        advantages_external = (returns - value_rollout)
        advantages_internal = (intrinsic - intrinsic_rollout)
        advantages_internal = (advantages_internal - advantages_internal.mean()) / (advantages_internal.std() + self.eps)
        weights = advantages_external + advantages_internal
        ratio = torch.exp(logp - fixed_logp)
        surrogate1 = ratio * weights
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * weights
        rnd_x = torch.zeros(1) if 'rnd_x' not in trajection else trajection['rnd_x']
        rnd_y = torch.zeros(1) if 'rnd_y' not in trajection else trajection['rnd_y']
        return (surrogate1, 
                surrogate2, 
                value_rollout,
                returns,
                intrinsic_rollout,
                intrinsic,
                rnd_x,
                rnd_y)
    
    def _get_loss(self, long_seq):
        loss = - torch.min(long_seq[0], long_seq[1]).sum() \
               + F.smooth_l1_loss(long_seq[2], long_seq[3], reduction='sum') \
               + F.smooth_l1_loss(long_seq[4], long_seq[5], reduction='sum') \
               + F.mse_loss(long_seq[6], long_seq[7], reduction='sum')
        return loss
    
    def forward(self, eps_trajection, inner_loop=0):
        self.agent.train()
        
        # Get the fixed referenced policy
        if inner_loop == 0:
            self.fixed_log_prob = []
            for i in eps_trajection:
                self.fixed_log_prob.append(self._fixed_log_prob(i))
                
        # Concate to long sequence
        long_seq = [[] for _ in range(8)]
        for trajection, fixed_logp in zip(eps_trajection, self.fixed_log_prob):
            for i, t in enumerate(self._process_single(trajection, fixed_logp)):
                long_seq[i].append(t)
        long_seq = [torch.cat(i) for i in long_seq]
        
        loss = self._get_loss(long_seq)
        return loss / len(eps_trajection)

