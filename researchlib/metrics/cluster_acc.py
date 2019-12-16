from collections import namedtuple
from scipy.optimize import linear_sum_assignment
import numpy as np


class ClusterAcc(namedtuple('ClusterAcc', [])):
    def __call__(self, x, y):
        x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()
        
        x = x.astype(np.int)
        y = y.astype(np.int)
        D = max(x.max(), y.max())+1
        w = np.zeros((D,D), dtype=np.int)
        for i in range(y.size):
            w[x[i], y[i]] += 1
        r_ind, c_ind = linear_sum_assignment(w.max() - w)
        return w[r_ind, c_ind].sum()/ y.size