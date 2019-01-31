from torch import nn
import torch

class MGULayer(nn.Module):
    def __init__(self, in_dim, out_dim, return_sequences=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rs = return_sequences
        self.W_f = nn.Parameter(torch.Tensor(in_dim + out_dim, out_dim))
        self.b_f = nn.Parameter(torch.Tensor(out_dim))
        self.W_h = nn.Parameter(torch.Tensor(in_dim + out_dim, out_dim))
        self.b_h = nn.Parameter(torch.Tensor(out_dim))
        
        for i in self.parameters():
            i.data.uniform_(-0.1, 0.1)
        
    def forward(self, x):
        '''
            Minimal Gated Unit
            
            f_t = sigmoid(W_f[h_(t-1), x_t] + b_f)
            h_tilda = tanh(W_h[f_t * h_(t-1), x_t] + b_h)
            h_t = (1 - f_t) * h_(t-1) + f_t * h_tilda
        '''
        
        x = x.permute(0, 2, 1)
        # x: (bs, features, ns)
        h_t = torch.zeros(x.size(0), self.out_dim)
        if next(self.parameters()).is_cuda:
            h_t = h_t.cuda()
        
        out = []
        for i in range(x.size(1)):
            f_t = torch.sigmoid((torch.cat((h_t, x[:, i, :]), dim=1) @ self.W_f) + self.b_f)
            h_tilda = torch.tanh((torch.cat(((f_t * h_t), x[:, i, :]), dim=1) @ self.W_h) + self.b_h)
            h_t = (1 - f_t) * h_t + f_t * h_tilda
            out.append(h_t)
        out = torch.stack(out, dim=1)
        out = out.permute(0, 2, 1)
        
        if self.rs:
            return out
        else:
            return out[:, :, -1]
            