from .ensemble import loss_ensemble
from .mapping import *

#from .gan import GANLoss

from .vae import VAELoss
from .likelihood import SmoothNLLoss, SmoothNLLLoss, NLLoss, NLLLoss, BCELoss, SNLLoss
from .dice import DiceLoss, FlattenDiceLoss
from .logcosh import LogCoshLoss
from .huber import HuberLoss
from .quantile import QuantileLoss
from .ssim_msssim import SSIMLoss, MSSSIMLoss
from .adaptive_robust_loss import AdaptiveRobustLoss
from .soft_dtw import SoftDTWLoss
from .distance import L1Loss, L2Loss
from .sparsemax import SparsemaxLoss
from .center import CenterLoss


from .cluster import Cluster
from .segmentation import Segmentation
from .noisy_classification import NoisyClassification



class Loss(object):
    # Submodules
    Cluster = Cluster
    Segmentation = Segmentation
    NoisyClassification = NoisyClassification
    
    # Ind
    L1 = L1Loss
    L2 = L2Loss
    MAE = L1Loss
    MSE = L2Loss
    NLL = NLLLoss
    NL = NLLoss
    BCE = BCELoss
    SNL = SNLLoss
    SmoothNL = SmoothNLLoss
    SmoothNLL = SmoothNLLLoss
    AdaptiveRobust = AdaptiveRobustLoss
    Dice = DiceLoss
    FlattenDice = FlattenDiceLoss
    LogCosh = LogCoshLoss
    Huber = HuberLoss
    Quantile = QuantileLoss
    SSIM = SSIMLoss
    MSSSIM = MSSSIMLoss
    SoftDTW = SoftDTWLoss
    Sparsemax = SparsemaxLoss
    VAE = VAELoss
    Center = CenterLoss
    
    
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
