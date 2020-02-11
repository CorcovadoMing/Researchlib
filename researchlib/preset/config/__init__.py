from .dawnfast import Dawnfast
from .stepwise import Stepwise, Stepwise300
from .poly import Poly
from .cosine import Cosine, Cosine300, CosineForRandWireModel
from .flat import Flat, Flat300


class Config(object):
    Dawnfast = Dawnfast
    Stepwise = Stepwise
    Stepwise300 = Stepwise300
    Poly = Poly
    Cosine = Cosine
    Cosine300 = Cosine300
    CosineForRandWireModel = CosineForRandWireModel
    Flat = Flat
    Flat300 = Flat300