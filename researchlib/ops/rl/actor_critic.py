from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import _discount_returns


class _ActorCritic(nn.Module):
    def __init__(self, agent, 
                 vf_coeff=0.5, 
                 int_coeff=1.0, 
                 rdn_coeff=1.0, 
                 state_node='state', 
                 policy_node='policy'):
        super().__init__()
        self.eps = np.finfo(np.float32).eps.item()
        self.vf_coeff = vf_coeff
        self.int_coeff = int_coeff
        self.rdn_coeff = rdn_coeff
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
            returns = torch.from_numpy(_discount_returns(trajection['reward'])).to(self.device).view(-1)
            intrinsic = torch.zeros_like(returns) if 'intrinsic' not in trajection else trajection['intrinsic']
            intrinsic = torch.from_numpy(_discount_returns(intrinsic)).to(self.device).view(-1)
        value_rollout = result['value'].view(-1)
        intrinsic_rollout = torch.zeros_like(intrinsic) if 'intrinsic' not in trajection \
                                                            else result['intrinsic'].view(-1)
        advantages_external = (returns - value_rollout)
        advantages_internal = (intrinsic - intrinsic_rollout)
        weights = advantages_external + advantages_internal
        weights = (weights - weights.mean()) / (weights.std() + self.eps)
        rnd_x = torch.zeros(1) if 'rnd_x' not in trajection else trajection['rnd_x']
        rnd_y = torch.zeros(1) if 'rnd_y' not in trajection else trajection['rnd_y']
        return (logp,
                weights,
                value_rollout,
                returns,
                intrinsic_rollout,
                intrinsic,
                rnd_x,
                rnd_y)
    
    def _get_loss(self, long_seq):
        loss = - (long_seq[0] * long_seq[1]).mean() \
               + F.mse_loss(long_seq[2], long_seq[3]) * self.vf_coeff \
               + F.mse_loss(long_seq[4], long_seq[5]) * self.int_coeff \
               + F.mse_loss(long_seq[6], long_seq[7]) * self.rdn_coeff
        return loss
    
    def forward(self, eps_trajection, inner_loop=0):
        self.agent.train()
        
        # Concate to long sequence
        long_seq = [[] for _ in range(8)]
        for trajection in eps_trajection:
            for i, t in enumerate(self._process_single(trajection)):
                long_seq[i].append(t)
        long_seq = [torch.cat(i) for i in long_seq]
        
        loss = self._get_loss(long_seq)
        return loss / len(eps_trajection)
