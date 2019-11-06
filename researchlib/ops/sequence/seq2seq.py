import torch
from torch import nn


class _Seq2Seq(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=1, max_length=100, attention=False):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.LSTM(in_dim, hidden, num_layers, batch_first=True)
        self.attention = attention
        attention_factor = 2 if attention else 1
        self.decoder_in_dim = hidden * attention_factor
        self.decoder = nn.LSTMCell(self.decoder_in_dim, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, out_dim + 1), 
                                 nn.LogSoftmax(-1))
        self.eos = out_dim
        self.max_length = max_length
    
    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        h, c = h.squeeze(0), c.squeeze(0)
        out = []
        inp = torch.zeros((x.size(0), self.decoder_in_dim)).to(x.device)
        for _ in range(self.max_length):
            h, c = self.decoder(inp, (h, c))
            inp = h
            pred = self.out(inp)
            out.append(pred)
            
            if self.attention:
                att = enc_out @ h.unsqueeze(-1)
                att = torch.softmax(att, 1)
                att = enc_out * att
                att = att.sum(1)
                inp = torch.cat([inp, att], dim=-1)
                
        result = torch.stack(out, dim=1)
        return result