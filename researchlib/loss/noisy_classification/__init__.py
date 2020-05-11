from .joint_optimization import JointOptimizationNLLoss
from .bootstrapping import SoftBootstrappingNLLoss, HardBootstrappingNLLoss

class NoisyClassification(object):
    JointOptimizationNL = JointOptimizationNLLoss
    SoftBootstrappingNL = SoftBootstrappingNLLoss
    HardBootstrappingNL = HardBootstrappingNLLoss