import torch
import numpy as np
from .pairwise import pairwise_distance


def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def lloyd(X, n_clusters, tol = 1e-4):
    X = torch.from_numpy(X).float().cuda()
    initial_state = forgy(X, n_clusters)

    while True:
        dis = pairwise_distance(X, initial_state)
        choice_cluster = torch.argmin(dis, dim = 1)
        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()
            selected = torch.index_select(X, 0, selected)
            initial_state[index] = selected.mean(dim = 0)

        center_shift = torch.sum(
            torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim = 1))
        )

        if center_shift ** 2 < tol:
            break

    return choice_cluster, initial_state


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, data):
        self.labels_, self.cluster_centers_ = lloyd(data, self.n_clusters)
        return self.labels_, self.cluster_centers_

    def fit_transform(self, data):
        self.labels_, self.cluster_centers_ = lloyd(data, self.n_cluster)
