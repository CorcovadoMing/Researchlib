import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class ElasticTransform:
    '''
    Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    '''
    def __init__(self):
        pass
    
    def __call__(self, x, choice):
        if choice:
            image = x.transpose(1,2,0)
            alpha = image.shape[0] * 0.5
            sigma = image.shape[0] * 0.05
            alpha_affine = image.shape[0] * 0.05

            random_state = np.random.RandomState(None)

            shape = image.shape
            shape_size = shape[:2]

            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dz = np.zeros_like(dx)

            _x, _y, _z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(_y+dy, (-1, 1)), np.reshape(_x+dx, (-1, 1)), np.reshape(_z, (-1, 1))

            image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
            x = image.transpose(2,0,1)
        return x
    
    def options(self):
        return [{'choice': b} for b in [True, False]]