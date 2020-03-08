from .layout import Layout
from .normalize import normalize
from .dynamic_normalize import DynamicNormalize
from .set_normalizer import set_normalizer
from .resize import Resize2d
from .bgr2rgb import BGR2RGB


class Preprocessing(object):
    Layout = Layout
    normalize = normalize
    DynamicNormalize = DynamicNormalize
    set_normalizer = set_normalizer
    Resize2d = Resize2d
    BGR2RGB = BGR2RGB