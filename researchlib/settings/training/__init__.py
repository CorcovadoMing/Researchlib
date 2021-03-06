from .dawnfast import Dawnfast
from .stepwise import Stepwise, Stepwise300
from .poly import Poly, Poly300
from .cosine import Cosine, Cosine300, CosineForRandWireModel
from .flat import Flat, Flat300
from .manual import Manual



class Training(object):
    Dawnfast = Dawnfast
    Stepwise = Stepwise
    Stepwise300 = Stepwise300
    Poly = Poly
    Poly300 = Poly300
    Cosine = Cosine
    Cosine300 = Cosine300
    CosineForRandWireModel = CosineForRandWireModel
    Flat = Flat
    Flat300 = Flat300
    Manual = Manual