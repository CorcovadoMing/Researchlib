from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from .utils import _discount


class _PPO(nn.Module):
    def __init__(self, agent, 
                 epsilon=0.2,
                 gae_ext_gamma=0.999,
                 gae_ext_lambda=0.95,
                 gae_int_gamma=0.99,
                 gae_int_lambda=0.95,
                 vf_coeff=0.5, 
                 int_coeff=2.0, 
                 rdn_coeff=1.0, 
                 state_node='state', 
                 policy_node='policy'):
        super().__init__()
        self.epsilon = epsilon
        self.eps = np.finfo(np.float32).eps.item()
        self.gae_ext_gamma = gae_ext_gamma
        self.gae_ext_lambda = gae_ext_lambda
        self.gae_int_gamma = gae_int_gamma
        self.gae_int_lambda = gae_int_lambda
        self.vf_coeff = vf_coeff
        self.int_coeff = int_coeff
        self.rdn_coeff = rdn_coeff
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
            
            weights = advantages_extrinsic + advantages_intrinsic
            weights = (weights - weights.mean()) / (weights.std() + self.eps)
            
        ratio = torch.exp(logp - fixed_logp.detach())
        surrogate1 = ratio * weights
        surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * weights
        intrinsic_loss = torch.zero_like(logp) if 'intrinsic_loss' not in trajection else trajection['intrinsic_loss']
        return (surrogate1, 
                surrogate2, 
                value_rollout,
                value_returns,
                intrinsic_rollout,
                intrinsic_returns,
                intrinsic_loss)
    
    def _get_loss(self, long_seq):
        loss = - torch.min(long_seq[0], long_seq[1]).mean() \
               + F.mse_loss(long_seq[2], long_seq[3]) * self.vf_coeff \
               + F.mse_loss(long_seq[4], long_seq[5]) * self.int_coeff \
               + long_seq[6].mean() * self.rdn_coeff
        return loss
    
    def forward(self, eps_trajection, inner_loop=0):
        self.agent.train()
        
        # Only support on-policy
        eps_trajection = eps_trajection[-1]
        
        # Get the fixed referenced policy
        if inner_loop == 0:
            self.fixed_log_prob = []
            for i in eps_trajection:
                self.fixed_log_prob.append(self._fixed_log_prob(i))
                
        # Concate to long sequence
        long_seq = [[] for _ in range(7)]
        for trajection, fixed_logp in zip(eps_trajection, self.fixed_log_prob):
            for i, t in enumerate(self._process_single(trajection, fixed_logp)):
                long_seq[i].append(t)
        long_seq = [torch.cat(i) for i in long_seq]
        
        loss = self._get_loss(long_seq)
        return loss / len(eps_trajection)

