from ...dataset import Preprocessing


def _PrepareImage2d(resize = None, bgr2rgb = False):
    magic_mean, magic_std = (125.31, 122.95, 113.87), (62.99, 62.09, 66.70)
    _list = [
        Preprocessing.Resize2d(resize) if resize is not None else None,
        Preprocessing.BGR2RGB() if bgr2rgb else None,
        Preprocessing.set_normalizer('static', magic_mean, magic_std),
        Preprocessing.Layout('HWC', 'CHW'),
    ]
    return list(filter(None, _list))