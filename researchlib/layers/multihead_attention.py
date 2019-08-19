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
        self.norm1 = nn.LayerNorm(out_feature)
        self.norm2 = nn.LayerNorm(out_feature)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.projection = nn.Linear(out_feature, out_feature)

        self.ffn1 = nn.Conv1d(out_feature, out_feature * 4, 1)
        self.ffn2 = nn.Conv1d(out_feature * 4, out_feature, 1)
        self.activator = nn.ReLU()

    def forward(self, x):
        bs = x.size(0)
        ts = x.size(2)
        x = x.transpose(-1, -2)  # bs, ts, features
        q = self.qw(x).view(bs, ts, self.heads, self.inter_feature).transpose(
            1, 2)  # bs, head, ts, feature
        k = self.kw(x).view(bs, ts, self.heads,
                            self.inter_feature).transpose(1, 2)
        v = self.vw(x).view(bs, ts, self.heads,
                            self.inter_feature).transpose(1, 2)
        e = (self.softmax(
            (q @ k.transpose(-1, -2)) /
            self.inter_feature**0.5)) @ v  # (ts, ts) @ (ts, feature)
        out = e.transpose(1, 2).contiguous().view(bs, ts, self.out_feature)
        out_inter = self.norm1(out + x)  # bs, ts, features
        out = self.drop1(out_inter)
        out = self.projection(out).transpose(-1, -2)  # bs, features, ts
        out = self.activator(self.ffn1(out))
        out = self.ffn2(out)
        out = self.norm2(out.transpose(-1, -2) + out_inter)
        out = self.drop2(out)
        return out.transpose(-1, -2)
