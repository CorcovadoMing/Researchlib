from sklearn import metrics
import numpy as np
from collections import namedtuple


class NMI(namedtuple('NMI', [])):
    def __call__(self, x, y):
        x, y = x.detach().cpu().numpy(), y.detach().view_as(x).cpu().numpy()
        return metrics.normalized_mutual_info_score(y, x, average_method='arithmetic')