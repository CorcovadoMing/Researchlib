import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
from torch.nn import Parameter as P
import warnings
warnings.simplefilter('ignore', UserWarning)
from .matrix import Matrix


class WrapInception(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                      requires_grad=False)
        self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                     requires_grad=False)

    def forward(self, x):
        x = (x + 1.) / 2.0
        x = (x - self.mean) / self.std
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=True)
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        logits = self.net.fc(
            F.dropout(pool, training=False).view(pool.size(0), -1))
        return pool, logits

import time
s = time.time()
class _InceptionModelV3:
    _model = None

    @staticmethod
    def lazy_load():
        if _InceptionModelV3._model is None:
            _InceptionModelV3._model = inception_v3(pretrained=True, transform_input=False)
            _InceptionModelV3._model = WrapInception(inception_model.eval()).cuda()

class InceptionScore(Matrix):
    def __init__(self):
        super().__init__()
        self.data = None
        _InceptionModelV3.lazy_load()

    def forward_g(self, x):
        with torch.no_grad():
            #(fake, real)
            x = x[0]
            _, x = _InceptionModelV3._model(x)
            x = F.softmax(x, dim=-1).data.cpu()
            if self.data is None:
                self.data = x
            else:
                self.data = torch.cat([self.data, x], dim=0)

    def output(self):
        part = self.data.numpy()
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        return {'inception_score': np.exp(np.mean(scores))}

    def reset(self):
        self.data = None


def sqrt_newton_schulz(A, numIters, dtype=None):
    with torch.no_grad():
        if dtype is None: dtype = A.type()
        batchSize = A.shape[0]
        dim = A.shape[1]
        normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
        Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
        I = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1,
                                                         1).type(dtype)
        Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batchSize, 1,
                                                         1).type(dtype)
        for i in range(numIters):
            T = 0.5 * (3.0 * I - Z.bmm(Y))
            Y = Y.bmm(T)
            Z = T.bmm(Z)
        sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    return sA


def torch_cov(m, rowvar=False):
    if m.dim() > 2: raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2: m = m.view(1, -1)
    if not rowvar and m.size(0) != 1: m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

    diff = mu1 - mu2
    # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
    covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()
    out = (diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) -
           2 * torch.trace(covmean))
    return out


import numpy as np
from scipy import linalg
import warnings


def calc_fid(mn1, cov1, mn2, cov2, eps=1e-6):
    mn1 = np.atleast_1d(mn1)
    mn2 = np.atleast_1d(mn2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    diff = mn1 - mn2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn(("fid() got singular product; adding {} to diagonal of "
                       "cov estimates").format(eps))
        offset = np.eye(d) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(cov1) + np.trace(cov2) - 2 * tr_covmean


class FID(Matrix):
    def __init__(self):
        super().__init__()
        self.fake_pool = None
        self.real_pool = None
        _InceptionModelV3.lazy_load()

    def forward_g(self, x):
        with torch.no_grad():
            fake_pool, _ = _InceptionModelV3._model(x[0])
            real_pool, _ = _InceptionModelV3._model(x[1])
            fake_pool = fake_pool.cpu()
            real_pool = real_pool.cpu()

            if self.real_pool is None:
                self.real_pool = real_pool
            else:
                self.real_pool = torch.cat([self.real_pool, real_pool], dim=0)

            if self.fake_pool is None:
                self.fake_pool = fake_pool
            else:
                self.fake_pool = torch.cat([self.fake_pool, fake_pool], dim=0)

    def output(self):
        real_mu, real_sigma = torch.mean(self.real_pool,
                                         0), torch_cov(self.real_pool,
                                                       rowvar=False)
        fake_mu, fake_sigma = torch.mean(self.fake_pool,
                                         0), torch_cov(self.fake_pool,
                                                       rowvar=False)
        #fid = torch_calculate_frechet_distance(fake_mu, fake_sigma, real_mu, real_sigma).numpy()
        fid = calc_fid(fake_mu.numpy(), fake_sigma.numpy(), real_mu.numpy(),
                       real_sigma.numpy())
        return {'fid': fid}

    def reset(self):
        self.fake_pool = None
        self.real_pool = None
