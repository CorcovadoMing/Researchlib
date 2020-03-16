from .lovasz import LovaszSoftmaxLoss
from .tverskey import BinaryTverskyLoss, FocalBinaryTverskyLoss, MultiTverskyLoss
from .focal import BinaryFocalLoss
from .dice import BinaryDiceLoss, DiceLoss, WBCELoss, WBCEDiceLoss


class Segmentation(object):
    LovaszSoftmax = LovaszSoftmaxLoss
    BinaryTversky = BinaryTverskyLoss
    FocalBinaryTversky = FocalBinaryTverskyLoss
    MultiTversky = MultiTverskyLoss
    BinaryFocal = BinaryFocalLoss
    BinaryDice = BinaryDiceLoss
    Dice = DiceLoss
    WBCE = WBCELoss
    WBCEDice = WBCEDiceLoss
    