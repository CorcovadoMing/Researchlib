from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import _discount


class _ActorCritic(nn.Module):
    def __init__(self, agent,
                 gae_ext_gamma=0.999,
                 gae_ext_lambda=0.95,
                 gae_int_gamma=0.99,
                 gae_int_lambda=0.95,
                 disable_int = False,
                 disable_ext = False,
                 int_weight = 1,
                 ext_weight = 1,
                 ext_coeff=0.5, 
                 int_coeff=2.0, 
                 int_fit_coeff=1.0, 
                 state_node='state', 
                 policy_node='policy'):
        super().__init__()
        self.eps = np.finfo(np.float32).eps.item()
        self.gae_ext_gamma = gae_ext_gamma
        self.gae_ext_lambda = gae_ext_lambda
        self.gae_int_gamma = gae_int_gamma
        self.gae_int_lambda = gae_int_lambda
        self.int_weight = int_weight
        self.ext_weight = ext_weight
        self.ext_coeff = ext_coeff
        self.int_coeff = int_coeff
        self.int_fit_coeff = int_fit_coeff
        self.agent = agent
        self.state_node = state_node
        self.policy_node = policy_node
        self.device = None
        
        if disable_int:
            self.int_weight = 0
            self.int_coeff = 0
            self.int_fit_coeff = 0
        
        if disable_ext:
            self.ext_weight = 0
            self.ext_coeff = 0
            
        
    def _process_single(self, trajection):
        if self.device is None:
            self.device = next(self.agent.parameters()).device
        result = self.agent({self.state_node: torch.stack(trajection['state'], 0).to(self.device)})
        action = torch.LongTensor(trajection['action']).to(self.device)
        logp = result[self.policy_node].log_prob(action).view(-1)
        value_rollout = result['value'].view(-1)
        intrinsic_rollout = torch.zeros_like(logp) if 'intrinsic' not in trajection \
                                                            else result['intrinsic'].view(-1)
        with torch.no_grad():
            value_returns = torch.from_numpy(_discount(trajection['reward'], self.gae_ext_gamma)).to(self.device).view(-1)
            intrinsic = torch.zeros_like(returns) if 'intrinsic' not in trajection else trajection['intrinsic']
            intrinsic_returns = torch.from_numpy(_discount(intrinsic, self.gae_int_gamma)).to(self.device).view(-1)
        
            advantages_extrinsic = torch.FloatTensor(trajection['reward']).to(self.device) \
                                  + self.gae_ext_gamma * torch.cat([value_rollout[1:], torch.zeros(1).to(self.device)]) \
                                  - value_rollout
            advantages_extrinsic = torch.from_numpy(_discount(advantages_extrinsic, 
                                                             self.gae_ext_gamma * self.gae_ext_lambda)).to(self.device)

            advantages_intrinsic = intrinsic.detach().to(self.device) \
                                  + self.gae_int_gamma * torch.cat([intrinsic_rollout[1:], torch.zeros(1).to(self.device)]) \
                                  - intrinsic_rollout
            advantages_intrinsic = torch.from_numpy(_discount(advantages_intrinsic, 
                                                             self.gae_int_gamma * self.gae_int_lambda)).to(self.device)
            
            weights = self.ext_weight * advantages_extrinsic + self.int_weight * advantages_intrinsic
            weights = (weights - weights.mean()) / (weights.std() + self.eps)
            
        intrinsic_loss = torch.zero_like(logp) if 'intrinsic_loss' not in trajection else trajection['intrinsic_loss']
        return (logp,
                weights,
                value_rollout,
                value_returns,
                intrinsic_rollout,
                intrinsic_returns,
                intrinsic_loss)
    
    def _get_loss(self, long_seq):
        loss = - (long_seq[0] * long_seq[1]).mean() \
               + F.mse_loss(long_seq[2], long_seq[3]) * self.ext_coeff \
               + F.mse_loss(long_seq[4], long_seq[5]) * self.int_coeff \
               + long_seq[6].mean() * self.int_fit_coeff
        return loss
    
    def forward(self, eps_trajection, inner_loop=0):
        self.agent.train()
        
        # Only support on-policy
        eps_trajection = eps_trajection[-1]
        
        # Concate to long sequence
        long_seq = [[] for _ in range(7)]
        for trajection in eps_trajection:
            for i, t in enumerate(self._process_single(trajection)):
                long_seq[i].append(t)
        long_seq = [torch.cat(i) for i in long_seq]
        
        loss = self._get_loss(long_seq)
        return loss / len(eps_trajection)
