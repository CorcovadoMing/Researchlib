from .augmentation import NumpyAugmentation, TorchAugmentation
from .preprocessing import NumpyPreprocessing, TorchPreprocessing

class template(object):
    NumpyAugmentation = NumpyAugmentation
    TorchAugmentation = TorchAugmentation
    NumpyPreprocessing = NumpyPreprocessing
    TorchPreprocessing = TorchPreprocessing