from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import _discount


class _REINFORCE(nn.Module):
    def __init__(self, agent, 
                 gae_ext_gamma=0.999,
                 gae_ext_lambda=0.95,
                 gae_int_gamma=0.99,
                 gae_int_lambda=0.95,
                 int_coeff=2.0, 
                 rdn_coeff=1.0, 
                 state_node='state', 
                 policy_node='policy'):
        super().__init__()
        self.eps = np.finfo(np.float32).eps.item()
        self.gae_ext_gamma = gae_ext_gamma
        self.gae_ext_lambda = gae_ext_lambda
        self.gae_int_gamma = gae_int_gamma
        self.gae_int_lambda = gae_int_lambda
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
        intrinsic_rollout = torch.zeros_like(logp) if 'intrinsic' not in trajection \
                                                            else result['intrinsic'].view(-1)
        with torch.no_grad():
            value_returns = torch.from_numpy(_discount(trajection['reward'], self.gae_ext_gamma)).to(self.device).view(-1)
            intrinsic = torch.zeros_like(returns) if 'intrinsic' not in trajection else trajection['intrinsic']
            intrinsic_returns = torch.from_numpy(_discount(intrinsic, self.gae_int_gamma)).to(self.device).view(-1)
        
            advantages_extrinsic = value_returns

            advantages_intrinsic = intrinsic.detach().to(self.device) \
                                  + self.gae_int_gamma * torch.cat([intrinsic_rollout[1:], torch.zeros(1).to(self.device)]) \
                                  - intrinsic_rollout
            advantages_intrinsic = torch.from_numpy(_discount(advantages_intrinsic, 
                                                             self.gae_int_gamma * self.gae_int_lambda)).to(self.device)
            
            weights = advantages_extrinsic + advantages_intrinsic
            weights = (weights - weights.mean()) / (weights.std() + self.eps)
            
        rnd_x = torch.zeros(1) if 'rnd_x' not in trajection else trajection['rnd_x']
        rnd_y = torch.zeros(1) if 'rnd_y' not in trajection else trajection['rnd_y']
        return (logp,
                weights,
                intrinsic_rollout,
                intrinsic_returns,
                rnd_x,
                rnd_y)
    
    def _get_loss(self, long_seq):
        loss = - (long_seq[0] * long_seq[1]).mean() \
               + F.mse_loss(long_seq[2], long_seq[3]) * self.int_coeff \
               + F.mse_loss(long_seq[4], long_seq[5]) * self.rdn_coeff
        return loss
    
    def forward(self, eps_trajection, inner_loop=0):
        self.agent.train()
        
        # Only support on-policy
        eps_trajection = eps_trajection[-1]
        
        # Concate to long sequence
        long_seq = [[] for _ in range(6)]
        for trajection in eps_trajection:
            for i, t in enumerate(self._process_single(trajection)):
                long_seq[i].append(t)
        long_seq = [torch.cat(i) for i in long_seq]
        
        loss = self._get_loss(long_seq)
        return loss / len(eps_trajection)

