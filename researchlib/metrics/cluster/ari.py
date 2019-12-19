from sklearn import metrics
import numpy as np
from collections import namedtuple


class ARI(namedtuple('ARI', [])):
    def __call__(self, x, y):
        x, y = x.detach().cpu().numpy(), y.detach().view_as(x).cpu().numpy()
        return metrics.adjusted_rand_score(y, x)
