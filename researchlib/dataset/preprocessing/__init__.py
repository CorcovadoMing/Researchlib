from .layout import Layout
from .normalize import normalize
from .dynamic_normalize import DynamicNormalize
from .set_normalizer import set_normalizer


class Preprocessing(object):
    Layout = Layout
    normalize = normalize
    DynamicNormalize = DynamicNormalize
    set_normalizer = set_normalizer