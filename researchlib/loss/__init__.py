from .ensemble import loss_ensemble
from .mapping import *

#from .gan import GANLoss
#from .vae import VAELoss

from .dice import DiceLoss, FlattenDiceLoss
from .logcosh import LogCoshLoss
from .huber import HuberLoss
from .quantile import QuantileLoss
from .ssim_msssim import SSIMLoss, MSSSIMLoss
from .adaptive_robust_loss import AdaptiveRobustLoss
from .soft_dtw import SoftDTWLoss

class loss(object):
    NegativeLogLikelyhood = nll_loss
    NegativeLikelyhood = nl_loss
    SmoothNegativeLogLikelyhood = smooth_nll_loss
    AdaptiveRobustLoss = AdaptiveRobustLoss
    DiceLoss = DiceLoss
    FlattenDiceLoss = FlattenDiceLoss
    LogCoshLoss = LogCoshLoss
    HuberLoss = HuberLoss
    QuantileLoss = QuantileLoss
    SSIMLoss = SSIMLoss
    MSSSIMLoss = MSSSIMLoss
    SoftDTWLoss = SoftDTWLoss
    
#     'bce': F.binary_cross_entropy
#     'adaptive_robust': AdaptiveRobustLoss(1),  # need to fix dimensions
#     'focal': FocalLoss(),  # need to be refined
#     'adaptive_focal': AdaptiveFocalLoss(),  # Experimental
#     'margin': MarginLoss(),
#     'mse': F.mse_loss,
#     'l2': F.mse_loss,
#     'mae': F.l1_loss,
#     'l1': F.l1_loss,
#     'huber': HuberLoss(),
#     'logcosh': LogCoshLoss
