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
from .acc import Acc
from .ece import ECE
from .l1 import L1
from .l2 import L2
from .hitrate import Hitrate
from .cluster_acc import ClusterAcc

class Metrics(object):
    Categorical = Categorical
    MetaCategorical = MetaCategorical
    Acc = Acc
    ClusterAcc = ClusterAcc
    ECE = ECE
    L1 = L1
    L2 = L2
    Hitrate = Hitrate