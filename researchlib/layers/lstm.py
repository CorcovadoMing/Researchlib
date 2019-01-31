from torch import nn

class LSTMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, return_sequences=False): 
        super().__init__() 
        self.f = nn.LSTM(in_dim, out_dim, batch_first=True)
        self.return_sequences = return_sequences
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.f(x)
        out = out.permute(0, 2, 1)
        if self.return_sequences:
            return out
        else:
            return out[:, :, -1]
    