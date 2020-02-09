from .dawnfast import Dawnfast
from .stepwise import Stepwise
from .poly import Poly
from .cosine import Cosine, CosineForRandWireModel


class Config(object):
    Dawnfast = Dawnfast
    Stepwise = Stepwise
    Poly = Poly
    Cosine = Cosine
    CosineForRandWireModel = CosineForRandWireModel