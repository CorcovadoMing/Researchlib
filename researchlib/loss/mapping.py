from torch import nn
import torch.nn.functional as F
from .custom_loss import *
from ..metrics import *
import torch
from .margin import *


def nl_loss(x, y): 
    return -x[range(y.shape[0]), y].log().mean()


def loss_mapping(loss_fn):
    # Assign loss function
    if loss_fn == 'nll':
        loss_fn = F.nll_loss
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = True
        default_metrics = Acc()
    
    elif loss_fn == 'nl':
        loss_fn = nl_loss
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = True
        default_metrics = Acc()

    elif loss_fn == 'bce':
        loss_fn = F.binary_cross_entropy
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = True
        #default_metrics = BCEAcc()
        default_metrics = AUROC()

    elif loss_fn == 'crossentropy':
        loss_fn = nn.CrossEntropyLoss()
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = True
        default_metrics = Acc()

    elif loss_fn == 'focal':
        loss_fn = FocalLoss()
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = True
        default_metrics = Acc()

    elif loss_fn == 'adaptive_focal':
        loss_fn = AdaptiveFocalLoss()
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = True
        default_metrics = Acc()

    elif loss_fn == 'margin':
        loss_fn = MarginLoss()
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = True
        default_metrics = Acc()

    elif loss_fn == 'mse' or loss_fn == 'l2':
        loss_fn = F.mse_loss
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        default_metrics = None

    elif loss_fn == 'mae' or loss_fn == 'l1':
        loss_fn = F.l1_loss
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        default_metrics = None

    elif loss_fn == 'huber':
        loss_fn = HuberLoss()
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        default_metrics = None

    elif loss_fn == 'logcosh':
        loss_fn = LogCoshLoss
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        default_metrics = None

    else:
        loss_fn = loss_fn
        require_long_ = False
        keep_x_shape_ = True
        keep_y_shape_ = True
        default_metrics = None

    return loss_fn, [require_long_], [keep_x_shape_], [keep_y_shape_
                                                       ], default_metrics
