import torch
from torch import nn


class _Seq2Seq(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=1, max_length=100):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)
        self.decoder = nn.LSTMCell(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, out_dim + 1), 
                                 nn.LogSoftmax(-1))
        self.eos = out_dim
        self.max_length = max_length
    
    def forward(self, x):
        _, (h, c) = self.encoder(x)
        h, c = h.squeeze(0), c.squeeze(0)
        out = []
        inp = torch.zeros((x.size(0), self.hidden)).to(x.device)
        for _ in range(self.max_length):
            h, c = self.decoder(inp, (h, c))
            inp = h
            pred = self.out(inp)
            out.append(pred)
        result = torch.stack(out, dim=1)
        return result