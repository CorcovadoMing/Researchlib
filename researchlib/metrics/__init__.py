# from .mape import MAPE
# from .maape import MAAPE
# from .dice import *
# from .correlation import *
# from .inception_score import *
# from .auroc import AUROC
# from .confusion_matrix import *
# from .bce_acc import BCEAcc
# from .kappa import Kappa
# from .meta_acc import MetaAcc

from .categorical import Categorical
from .meta_categorical import MetaCategorical
from .binary_categorical import BinaryCategorical
from .acc import Acc
from .ece import ECE
from .l1 import L1
from .l2 import L2
from .hitrate import Hitrate
from .cluster import Cluster
from .rl import RL
from .psnr import PSNR
from .aum import AUM


class Metrics(object):
    Cluster = Cluster
    RL = RL
    
    Categorical = Categorical
    MetaCategorical = MetaCategorical
    BinaryCategorical = BinaryCategorical
    
    Acc = Acc
    ECE = ECE
    L1 = L1
    L2 = L2
    Hitrate = Hitrate
    PSNR = PSNR
    AUM = AUM