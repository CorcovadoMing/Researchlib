from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, in_feature, out_feature, heads=8):
        super().__init__()
        
        assert out_feature % heads == 0
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.heads = heads
        self.inter_feature = int(out_feature / heads)
        
        self.qw = nn.Linear(in_feature, out_feature)
        self.kw = nn.Linear(in_feature, out_feature)
        self.vw = nn.Linear(in_feature, out_feature)
        self.softmax = nn.Softmax(-1)
        self.norm = nn.LayerNorm(out_feature)
        
    def forward(self, x):
        bs = x.size(0)
        ts = x.size(2)
        x = x.transpose(-1, -2) # bs, ts, features
        q = self.qw(x).view(bs, ts, self.heads, self.inter_feature).transpose(1, 2)
        k = self.kw(x).view(bs, ts, self.heads, self.inter_feature).transpose(1, 2)
        v = self.vw(x).view(bs, ts, self.heads, self.inter_feature).transpose(1, 2)
        e = self.softmax((q @ k.transpose(-1, -2)) / self.inter_feature**0.5) @ v
        out = e.transpose(1, 2).contiguous().view(bs, ts, self.out_feature)
        out = self.norm(out + x).transpose(-1, -2)
        return out