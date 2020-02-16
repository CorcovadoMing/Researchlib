import torch
from torch import nn
# from torch_scatter import scatter_add
# from torch_geometric.utils import softmax


class _Set2Set(nn.Module):
    def __init__(self, in_channels, processing_steps, num_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers, batch_first=True)

    def forward(self, x):
        batch_size = x.size(0)

        h = x.new_zeros((self.num_layers, batch_size, self.in_channels))
        c = x.new_zeros((self.num_layers, batch_size, self.in_channels))
        q_star = x.new_zeros(batch_size, x.size(1), self.out_channels)

        for i in range(self.processing_steps):
            q, (h, c) = self.lstm(q_star, (h, c))
            e = (x * q[torch.arange(x.size(0))]).sum(dim=-1, keepdim=True)
            a = softmax(e, torch.arange(x.size(0)), num_nodes=batch_size)
            r = scatter_add(a * x, torch.arange(x.size(0)), dim=0, dim_size=batch_size)
            q_star = torch.cat([q, r], dim=-1)
            
        return q_star