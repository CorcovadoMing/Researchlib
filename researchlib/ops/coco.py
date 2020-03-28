from torch import nn
import torch


class _COCO(nn.Module):
    '''
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
    '''

    def __init__(self, num_classes, feat_dim, alpha=6.25):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha * nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))

        return logits