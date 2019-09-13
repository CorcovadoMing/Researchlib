from .template.block import _Block
from .unit import unit
import torch
from torch import nn


class _GCNBlock(_Block):
    def __postinit__(self):
        A = self._get_param('adjacency_matrix', None, required=True)
        coff_embedding = self._get_param('coff_embedding', 4)
        subset = self._get_param('subset', 3)
        
        self.hidden_dim = self.out_dim // coff_embedding
        self.num_subset = subset

        self.A = torch.from_numpy(A).float()
        self.B = nn.Parameter(torch.from_numpy(A).float())

        self.embedded_1 = nn.ModuleList()
        self.embedded_2 = nn.ModuleList()
        self.embedded_out = nn.ModuleList()
        
        for i in range(self.num_subset):
            self.embedded_1.append(self.op(self.in_dim, self.hidden_dim, 1))
            self.embedded_2.append(self.op(self.in_dim, self.hidden_dim, 1))
            self.embedded_out.append(self.op(self.in_dim, self.out_dim, 1))

        if self.in_dim != self.out_dim:
            self.down = nn.Sequential(
                self.op(self.in_dim, self.out_dim, 1),
                nn.BatchNorm2d(self.out_dim)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(self.out_dim)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()


    def forward(self, x):
        N, C, T, V = x.size()
        
        AB = self.A.to(x.device) + self.B

        y = None
        for i in range(self.num_subset):
            C1 = self.embedded_1[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.hidden_dim * T)
            C2 = self.embedded_2[i](x).view(N, self.inter_c * T, V)
            C1 = self.soft(torch.matmul(C1, C2) / C1.size(-1))  # N V V
            C1 = C1 + AB[i]
            C2 = x.view(N, C * T, V)
            z = self.embedded_out[i](torch.matmul(C2, C1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)