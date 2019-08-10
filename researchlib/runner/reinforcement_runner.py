from itertools import count
from torch.distributions import Categorical
import torchcontrib
import torch
import numpy as np
from torch.optim import *
import torch.nn.functional as F
import imageio


class ReinforcementRunner:
    def __init__(self, env, net, estimator):
        self.env = env
        self.net = net.cuda()
        self.estimator = estimator.cuda()
        self.estimator_optimizer = Adam(estimator.parameters(), 1e-3)
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
        self.log_policy_pool = []
        self.optimizer = torchcontrib.optim.SWA(Adam(net.parameters(), 1e-3))
        self.optimizer.swap_swa_sgd()

    def _forward(self, prev_state=None, collect=True):
        if prev_state is None:
            prev_state = self.env.reset()
        if collect: self.state_pool.append(prev_state)
        probs = self.net(torch.from_numpy(prev_state).float().cuda())
        distribution = Categorical(probs)
        action = distribution.sample()
        state, reward, done, _ = self.env.step(int(action))
        if collect:
            self.action_pool.append(action)
            self.reward_pool.append(reward)
            self.log_policy_pool.append(distribution.log_prob(action))
        return state, reward, distribution, action, done

    def _train(self, swa_update, gamma=0.99):
        self.net.train()
        self.optimizer.swap_swa_sgd()
        for i in reversed(range(len(self.reward_pool) - 1)):
            if self.reward_pool[i] != 0:
                self.reward_pool[i] += gamma * self.reward_pool[i + 1]
        self.reward_pool = torch.Tensor(self.reward_pool)
        self.reward_pool = (self.reward_pool - self.reward_pool.mean()) / (
            self.reward_pool.std() + 1e-7)

        baseline = self.estimator(
            torch.Tensor(self.state_pool).float().cuda()).cpu().squeeze()
        rewards = self.reward_pool - baseline
        rewards = rewards.detach()

        self.optimizer.zero_grad()
        loss = 0
        for lb, reward in zip(self.log_policy_pool, rewards):
            loss += -lb * reward
        loss /= len(rewards)
        loss.backward()
        self.optimizer.step()
        if swa_update:
            self.optimizer.update_swa()
        self.optimizer.swap_swa_sgd()

        self.estimator_optimizer.zero_grad()
        loss = F.mse_loss(baseline, self.reward_pool)
        loss.backward()
        self.estimator_optimizer.step()

    def optimize(self, num_episode=500, batch_size=5):
        for episode in range(1, num_episode):
            self.net.eval()
            state = None
            for _ in count(1):
                state, _, _, _, done = self._forward(state)
                if done:
                    self.reward_pool[-1] = 0
                    break

            if episode % batch_size == 0:
                swa_update = False  #TODO
                print(sum(self.reward_pool) / batch_size)
                self._train(swa_update)
                self.state_pool = []
                self.action_pool = []
                self.reward_pool = []
                self.log_policy_pool = []

    def inference(self):
        self.net.eval()
        state = None
        for iteration in count(1):
            state, reward, _, action, done = self._forward(state,
                                                           collect=False)
            self.env.render(
                str(iteration) + ' ' + str(action) + ' ' + str(reward))
            if done:
                break

    def make_gif(self, name, duration):
        self.net.eval()
        state = None
        buffer = []
        for iteration in count(1):
            state, reward, _, action, done = self._forward(state,
                                                           collect=False)
            buffer.append(self.env.render(None, return_cache=True))
            if done:
                break
        imageio.mimsave(name + '.gif', buffer, duration=duration)
