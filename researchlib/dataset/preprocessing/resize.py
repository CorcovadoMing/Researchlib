import cv2


class Resize2d:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, x):
        return cv2.resize(x, (self.size, self.size))