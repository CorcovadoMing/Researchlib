from .format_transpose import format_transpose
from .normalize import normalize
from .dynamic_normalize import DynamicNormalize
from .set_normalizer import set_normalizer


class preprocessing(object):
    format_transpose = format_transpose
    normalize = normalize
    DynamicNormalize = DynamicNormalize
    set_normalizer = set_normalizer