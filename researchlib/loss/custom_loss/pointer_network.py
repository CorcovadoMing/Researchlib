import torch
from torch import nn
from ...layers import *

class PointerNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target, lengths):
        """
        Args:
          logits : predicts (bz, tgt_max_len, src_max_len)
          target : label data (bz, tgt_max_len)
          lengths : length of label data (bz)
        """
        _, tgt_max_len = target.size()
        logits_flat = logits.view(-1, logits.size(-1))
        log_logits_flat = torch.log(logits_flat)
        target_flat = target.view(-1, 1)
        losses_flat = -torch.gather(log_logits_flat, dim=1, index = target_flat)
        losses = losses_flat.view(*target.size())
        mask = sequence_mask(lengths, tgt_max_len)
        mask = Variable(mask)
        losses = losses * mask.float()
        loss = losses.sum() / lengths.float().sum()
        return loss