from torch import nn

class LSTMLayer(nn.Module):
    def __init__(self, in_dim, out_dim, return_sequences=False): 
        super().__init__() 
        self.f = nn.LSTM(in_dim, out_dim)
        self.return_sequences = return_sequences
    
    def forward(self, x):
        if self.return_sequences:
            return self.f(x)[0]
        else:
            return self.f(x)[0][:, -1]
    