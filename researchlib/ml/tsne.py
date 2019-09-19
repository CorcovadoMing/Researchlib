from sklearn import manifold, datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import squareform
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

import torch
import torch.autograd
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn

import numpy as np
import random


def pairwise(data):
    n_obs, dim = data.size()
    xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
    xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
    dkl2 = ((xk - xl) ** 2.0).sum(2).squeeze()
    return dkl2


class VTSNE(nn.Module):
    def __init__(self, n_points, n_topics):
        super().__init__()
        self.n_points = n_points
        # Logit of datapoint-to-topic weight
        self.logits_mu = nn.Embedding(n_points, n_topics)
        self.logits_lv = nn.Embedding(n_points, n_topics)

    @property
    def logits(self):
        return self.logits_mu

    def reparametrize(self, mu, logvar):
        # From VAE example
        # https://github.com/pytorch/examples/blob/master/vae/main.py
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld = torch.sum(kld).mul_(-0.5)
        return z, kld

    def sample_logits(self, i = None):
        if i is None:
            return self.reparametrize(self.logits_mu.weight, self.logits_lv.weight)
        else:
            return self.reparametrize(self.logits_mu(i), self.logits_lv(i))

    def forward(self, pij, i, j):
        # Get  for all points
        x, loss_kldrp = self.sample_logits()
        # Compute squared pairwise distances
        dkl2 = pairwise(x)
        # Compute partition function
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        # Compute the numerator
        xi, _ = self.sample_logits(i)
        xj, _ = self.sample_logits(j)
        num = ((1. + (xi - xj) ** 2.0).sum(1)).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        # Compute KLD(pij || qij)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        # Compute sum of all variational terms
        return loss_kld.sum() + loss_kldrp.sum() * 1e-7

    def __call__(self, *args):
        return self.forward(*args)


def preprocess(x, perplexity = 30, metric = 'euclidean'):
    """ Compute pairiwse probabilities for MNIST pixels.
    """
    distances2 = pairwise_distances(x, metric = metric, squared = True)
    # This return a n x (n-1) prob array
    pij = manifold.t_sne._joint_probabilities(distances2, perplexity, False)
    # Convert to n x n prob array
    pij = squareform(pij)
    return pij


def plot(model, y = None):
    # Visualize the results
    embed = model.logits.weight.cpu().data.numpy()
    plt.figure()
    # Visualize with ellipses
    var = np.sqrt(model.logits_lv.weight.clone().exp_().cpu().data.numpy())
    ax = plt.gca()
    #for xy, (w, h), c in zip(embed, var, y):
    for xy, (w, h) in zip(embed, var):
        e = Ellipse(xy = xy, width = w, height = h, ec = None, lw = 0.0)
        #e.set_facecolor(plt.cm.Paired(c * 1.0 / y.max()))
        e.set_alpha(0.8)
        ax.add_artist(e)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-9, 9)
    plt.axis('off')
    plt.show()


def chunks(n, *args):
    """Yield successive n-sized chunks from l."""
    endpoints = list(range(0, args[0].size(0) - n, n))
    random.shuffle(endpoints)
    for start in endpoints:
        yield [a[start:start + n] for a in args]


class Wrapper():
    def __init__(self, model, cuda = True):
        self.cuda = cuda
        self.model = model
        if cuda:
            self.model.cuda()

    def fit(self, batch_size, lr, epochs, *args):
        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr = lr)
        for epoch in range(epochs):
            total = 0.0
            for datas in chunks(batch_size, *args):
                opt.zero_grad()
                loss = self.model(*datas)
                loss.backward()
                opt.step()
                total += loss.item()
            msg = 'Train Epoch: {} \tLoss: {:.6e}'
            msg = msg.format(epoch, total / (len(args[0]) * 1.0))
            print(msg)


class TSNE:
    def __init__(self, x):
        pij2d = preprocess(x)
        i, j = np.indices(pij2d.shape)
        i = i.ravel()
        j = j.ravel()
        pij = pij2d.ravel().astype('float32')
        # Remove self-indices
        idx = i != j
        i, j, pij = i[idx], j[idx], pij[idx]
        self.model = VTSNE(x.shape[0], 2)
        self.wrap = Wrapper(self.model)
        pij, i, j = torch.from_numpy(pij).cuda(), torch.from_numpy(i).cuda(
        ), torch.from_numpy(j).cuda()
        self.data = (pij, i, j)

    def fit(self, batch_size = 1024, lr = 5e-2, epochs = 50):
        self.wrap.fit(batch_size, lr, epochs, *self.data)

    def plot(self):
        plot(self.model)
