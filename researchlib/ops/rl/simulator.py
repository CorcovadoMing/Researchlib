from torch import nn
import torch


class _Simulator(nn.Module):
    def __init__(self, agent, env, state_node='state', action_node='action'):
        super().__init__()
        self.agent = agent
        self.env = env
        self.state_node = state_node
        self.action_node = action_node
    
    def forward(self, batch_size):
        trajection = {'state': [], 'action': [], 'reward': []}
        state, reward, done, _ = self.env.reset()
        eps_trajection = []
        self.agent.eval()
        while True:
            if not done:
                state = torch.from_numpy(state).float()
                result = self.agent({self.state_node: state[None, ...]})
                action = result[self.action_node].item()
                trajection['state'].append(state)
                trajection['action'].append(action)
                state, reward, done, _ = self.env.step(action)
                trajection['reward'].append(reward)
            else:
                eps_trajection.append(trajection)
                if len(eps_trajection) >= batch_size:
                    break
                else:
                    trajection = {'state': [], 'action': [], 'reward': []}
                    state, reward, done, _ = self.env.reset()
        return eps_trajection