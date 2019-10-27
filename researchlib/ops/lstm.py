from torch import nn


class _LSTM(nn.Module):
    def __init__(self, in_dim, out_dim, return_sequences = False, bidirection = False):
        super().__init__()
        self.ff = nn.LSTM(in_dim, out_dim, batch_first = True)
        self.return_sequences = return_sequences
        self.bidirection = bidirection
        if bidirection:
            self.bf = nn.LSTM(in_dim, out_dim, batch_first = True)

    def forward(self, x):
        out, _ = self.ff(x)
        if self.bidirection:
            ts = list(range(x.size(1)))[::-1]
            b_out, _ = self.bf(x[:, ts, :])
            out = torch.cat([out, b_out], dim = -1)

        if self.return_sequences:
            return out
        else:
            return out[:, -1, :]
