from torch import nn
import torch


class _Simulator(nn.Module):
    def __init__(self, agent, env, state_node='state', action_node='action', policy_node='policy'):
        super().__init__()
        self.agent = agent
        self.env = env
        self.state_node = state_node
        self.action_node = action_node
        self.policy_node = policy_node
        self.cache = None
        self.device = None
        self.enable = True
    
    def set_enable(self):
        self.enable = True
    
    def set_disable(self):
        self.enable = False
    
    def forward(self, batch_size):
        if self.enable:
            if self.device is None:
                self.device = next(self.agent.parameters()).device
            trajection = {'state': [], 'action': [], 'reward': [], 'policy': []}
            state, reward, done, _ = self.env.reset()
            eps_trajection = []
            self.agent.eval()
            while True:
                if not done:
                    state = torch.from_numpy(state).float()
                    result = self.agent({self.state_node: state[None, ...].to(self.device)})
                    action = result[self.action_node].item()
                    trajection['state'].append(state)
                    try:
                        trajection['policy'].append(result[self.policy_node])
                    except:
                        trajection['policy'].append(None)
                    trajection['action'].append(action)
                    state, reward, done, _ = self.env.step(action)
                    trajection['reward'].append(reward)
                else:
                    eps_trajection.append(trajection)
                    if len(eps_trajection) >= batch_size:
                        break
                    else:
                        trajection = {'state': [], 'action': [], 'reward': [], 'policy': []}
                        state, reward, done, _ = self.env.reset()
            self.cache = eps_trajection
        return self.cache