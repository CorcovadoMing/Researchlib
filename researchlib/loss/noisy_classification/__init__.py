from .joint_optimization import JointOptimizationNLLoss
from .bootstrapping import SoftBootstrappingNLLoss, HardBootstrappingNLLoss
from .symmetry import SymmetryNLLoss
from .improved_mae import IMAENLLoss


class NoisyClassification(object):
    JointOptimizationNL = JointOptimizationNLLoss
    SoftBootstrappingNL = SoftBootstrappingNLLoss
    HardBootstrappingNL = HardBootstrappingNLLoss
    SymmetryNL = SymmetryNLLoss
    IMAENL = IMAENLLoss