from torch import nn
import torch.nn.functional as F
import torch

class _SelfAttention2d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.empty(1).fill_(0), requires_grad=True)

    def forward(self, x):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return x + self.gamma * o