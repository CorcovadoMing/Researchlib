from .dawnfast import Dawnfast
from .stepwise import Stepwise, Stepwise300
from .poly import Poly
from .cosine import Cosine, CosineForRandWireModel


class Config(object):
    Dawnfast = Dawnfast
    Stepwise = Stepwise
    Stepwise300 = Stepwise300
    Poly = Poly
    Cosine = Cosine
    CosineForRandWireModel = CosineForRandWireModel