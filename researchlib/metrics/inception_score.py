import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import warnings
warnings.simplefilter('ignore', UserWarning)
from .matrix import *

inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
inception_model.eval()
    
class InceptionScore(Matrix):
    def __init__(self):
        super().__init__()
        self.inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
        self.inception_model.eval()
        self.data = None
    
    def forward_g(self, x):
        with torch.no_grad():
            x = x[0] # GAN output should be contained in the fist place
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
            x = inception_model(x)
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