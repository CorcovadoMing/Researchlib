from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, target_class=1, need_exp=False):
        super().__init__()
        self.smooth = smooth
        self.target_class = target_class
        self.need_exp = need_exp
    
    def forward(self, x, y):
        x = x[:, self.target_class, :, :]
        if self.need_exp:
            x = x.exp()
        x = x.view(x.size(0), -1)
        y = (y == self.target_class)
        y = y.view(x.size(0), -1).float()
        intersection = (x * y).sum(1)
        ratio = 2*(intersection + self.smooth) / (y.sum(1) + x.sum(1) + self.smooth)
        return 1-ratio.mean()