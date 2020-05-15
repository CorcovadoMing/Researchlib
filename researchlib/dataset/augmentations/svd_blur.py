import numpy as np
from collections import namedtuple


def randomized_svd(M, k=8):
    m, n = M.shape
    transpose = False
    if m < n:
        transpose = True
        M = M.T
        
    rand_matrix = np.random.normal(size=(M.shape[1], k))  # short side by k
    Q, _ = np.linalg.qr(M @ rand_matrix, mode='reduced')  # long side by k
    smaller_matrix = Q.T @ M                              # k by short side
    U_hat, s, V = np.linalg.svd(smaller_matrix, full_matrices=False)
    U = Q @ U_hat
    
    if transpose:
        return V.T, s.T, U.T
    else:
        return U, s, V
    
    
def _svd_blur(img, K):
    rec = []
    for i in range(img.shape[-1]):
        u, s, v = randomized_svd(img[..., i])
        s[..., -K:] = 0
        rec.append((u * s) @ v)
    return np.clip(np.stack(rec, -1), 0, 255).astype(np.uint8)


class SVDBlur(namedtuple('SVDBlur', ('img_size'))):
    def __call__(self, x, y, choice, K):
        return (_svd_blur(x, K), y) if choice else (x, y)

    def options(self, prob=0.5):
        return {
            'choice': np.random.choice([True, False], p=[prob, 1-prob], size=1),
            'K': np.random.choice(range(self.img_size - (self.img_size//16)))
        }



