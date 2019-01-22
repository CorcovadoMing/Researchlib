import torch
import torch.nn.functional as F
from torch import nn
from ..utils import *

class NeuralProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self.r_dim = 128
        self.z_dim = 128

        self.h_1 = nn.Linear(3, 400)
        self.h_2 = nn.Linear(400, 400)
        self.h_3 = nn.Linear(400, self.r_dim)

        self.r_to_z_mean = nn.Linear(self.r_dim, self.z_dim)
        self.r_to_z_logvar = nn.Linear(self.r_dim, self.z_dim)

        self.g_1 = nn.Linear(self.z_dim + 2, 400)
        self.g_2 = nn.Linear(400, 400)
        self.g_3 = nn.Linear(400, 400)
        self.g_4 = nn.Linear(400, 400)
        self.g_5 = nn.Linear(400, 1)
        
        self.x_grid = generate_grid(28, 28)

    def h(self, x_y):
        x_y = F.relu(self.h_1(x_y))
        x_y = F.relu(self.h_2(x_y))
        x_y = F.relu(self.h_3(x_y))
        return x_y

    def aggregate(self, r):
        return torch.mean(r, dim=1)

    def reparameterise(self, z):
        mu, logvar = z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(mu)
        z_sample = z_sample.unsqueeze(1).expand(-1, 784, -1)
        return z_sample

    def g(self, z_sample, x_target):
        z_x = torch.cat([z_sample, x_target], dim=2)
        input = F.relu(self.g_1(z_x))
        input = F.relu(self.g_2(input))
        input = F.relu(self.g_3(input))
        input = F.relu(self.g_4(input))
        y_hat = torch.sigmoid(self.g_5(input))
        return y_hat

    def xy_to_z_params(self, x, y):
        x_y = torch.cat([x, y], dim=2)
        r_i = self.h(x_y)
        r = self.aggregate(r_i)

        mu = self.r_to_z_mean(r)
        logvar = self.r_to_z_logvar(r)

        return mu, logvar

    def forward(self, x_context, y_context, y_all=None):
        x_all = self.x_grid.expand(x_context.size(0), -1, -1)
        
        z_context = self.xy_to_z_params(x_context, y_context)  # (mu, logvar) of z
        if self.training:  # loss function will try to keep z_context close to z_all
            z_all = self.xy_to_z_params(x_all, y_all)
        else:  # at test time we don't have the image so we use only the context
            z_all = z_context

        z_sample = self.reparameterise(z_all)

        # reconstruct the whole image including the provided context points
        x_target = self.x_grid.expand(y_context.shape[0], -1, -1)
        y_hat = self.g(z_sample, x_target)

        return y_hat, z_all, z_context
