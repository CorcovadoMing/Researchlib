from ..utils import _register_method
import torch

__methods__ = []
register_method = _register_method(__methods__)


@register_method
def predict(self, x, y = [], augmentor = None):
    with torch.no_grad():
        self.preload_gpu()
        try:
            guess = self.model(x.cuda())
            if augmentor:
                aug_list = augmentor.someof + augmentor.oneof
                for aug_fn in aug_list:
                    _x, _ = aug_fn(x.numpy(), y)
                    _x = torch.from_numpy(np.ascontiguousarray(_x))
                    guess = self.model(_x.cuda())
                    del _x
                guess /= len(aug_list)
            guess = guess.cpu()
        except:
            raise
        finally:
            del x, y
            self.unload_gpu()
    return guess
