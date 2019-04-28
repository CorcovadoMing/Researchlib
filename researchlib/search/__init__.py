from .single_point import _IterativeImprovement, _SimulatedAnnealing
from .population_based import _GeneticAlgorithm
from .benchmarks import *

class search(object):
    # Single-point-based
    II=_IterativeImprovement
    SA=_SimulatedAnnealing
    
    # Population-based
    GA=_GeneticAlgorithm