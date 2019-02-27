from torch import nn
import torch.nn.functional as F
from .custom_loss import *
from ..metrics import *

def loss_mapping(loss_fn):
    # Assign loss function
    if loss_fn == 'nll':
        loss_fn = F.nll_loss
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = False
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = Acc()
        
    elif loss_fn == 'bce':
        loss_fn = nn.BCELoss()
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = BCEAcc()
        
    elif loss_fn == 'crossentropy':
        loss_fn = nn.CrossEntropyLoss()
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = False
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = Acc()
        
    elif loss_fn == 'focal':
        loss_fn = FocalLoss()
        require_long_ = True
        keep_x_shape_ = False
        keep_y_shape_ = False
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = Acc()
        
        
    elif loss_fn == 'mse' or loss_fn == 'l2':
        loss_fn = F.mse_loss
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = None

    elif loss_fn == 'mae' or loss_fn == 'l1':
        loss_fn = F.l1_loss
        require_long_ = False
        keep_x_shape_ = False
        keep_y_shape_ = False
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = None
        
    else:
        loss_fn = loss_fn
        require_long_ = False
        keep_x_shape_ = True
        keep_y_shape_ = True
        require_data_ = False
        try:
            require_data_ = loss_fn.require_data
        except:
            pass
        default_metrics = None
    
    return loss_fn, require_long_, keep_x_shape_, keep_y_shape_, require_data_, default_metrics