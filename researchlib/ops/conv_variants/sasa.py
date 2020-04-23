import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _single, _pair, _triple
import math


class _SASA2d(nn.Module):
    '''
        TODO: 
            - relative embedding needs to support different size of kernels
            - multi-head attention
    '''
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride = 1,
        padding = 0,
        groups = 1,
        bias = False,
        **kwargs
    ):
        super().__init__()

        if kernel_size == 1:
            self._pass = True
            self._op = nn.Conv2d(in_dim, out_dim, 1, stride = stride)
        else:
            self._pass = False

            self.in_dim = in_dim
            self.out_dim = out_dim

            self.kernel_size = _pair(kernel_size)
            self.padding = _pair(padding)
            self.stride = _pair(stride)

            self.kernel_flatshape = torch.numel(torch.Tensor(*self.kernel_size))

            self.wq = nn.Conv2d(in_dim, out_dim, 1, bias = False)
            self.wk = nn.Conv2d(in_dim, out_dim, 1, bias = False)
            self.wv = nn.Conv2d(in_dim, out_dim, 1, bias = False)

            self.q_pos = math.ceil(self.kernel_flatshape / 2)

            self.relative_row_encoding = torch.nn.Embedding(self.kernel_size[0], int(out_dim / 2))
            self.relative_column_encoding = torch.nn.Embedding(
                self.kernel_size[1], int(out_dim / 2)
            )

            self.row = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]).long()
            self.column = torch.Tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]).long()

    def forward(self, x):
        if self._pass:
            return self._op(x)

        _, _, w, h = x.size()
        x = F.unfold(x, self.kernel_size, 1, self.padding,
                     self.stride).reshape(x.size(0), self.in_dim, self.kernel_flatshape, -1)
        q = self.wq(x)[:, :, self.q_pos, :].unsqueeze(2)
        k = self.wk(x)
        v = self.wv(x)

        embedded = torch.cat([
            self.relative_row_encoding(self.row.to(x.device)),
            self.relative_column_encoding(self.column.to(x.device))
        ],
                             dim = 1).transpose(-1, -2)
        embedded = embedded[None, :, :, None].expand_as(k).to(x.device)

        att = F.softmax(
            torch.einsum('bmik,bmjk -> bmij', q, k) +
            torch.einsum('bmik,bmjk -> bmij', q, embedded), -1
        )
        return torch.einsum('bmij,bmjk -> bmjk', att, v).sum(2).reshape(
            x.size(0), self.out_dim, int(w / self.stride[0]), int(h / self.stride[1])
        )
