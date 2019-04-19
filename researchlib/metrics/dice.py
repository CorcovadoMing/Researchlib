from .matrix import *
from ..utils import *

import numpy as np
import matplotlib.pyplot as plt
import torch

class DiceMatrix(Matrix):
    def __init__(self, smooth=1, target_class=1, need_exp=False):
        super().__init__()
        self.smooth = smooth
        self.target_class = target_class
        self.need_exp = need_exp
        self.records = []
        
    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup
            #y_pred, y_true, y_true_res, lam = loss_input[0].cpu(), loss_input[1].cpu(), loss_input[2].cpu(), loss_input[3]
            #_, predicted = torch.max(y_pred, 1)
            #self.correct += (lam * predicted.eq(y_true).sum().float()
            #            + (1 - lam) * predicted.eq(y_true_res).sum().float()).numpy()
            
            y_pred, y_true = loss_input[0].cpu(), loss_input[1].cpu()
            
            y_pred = y_pred[:, self.target_class, :, :]
            if self.need_exp:
                y_pred = y_pred.exp()
            y_pred = y_pred.view(y_pred.size(0), -1)
            y_true = (y_true == self.target_class)
            y_true = y_true.view(y_pred.size(0), -1).float()
            intersection = (y_pred * y_true).sum(1)
            ratio = 2*(intersection + self.smooth) / (y_true.sum(1) + y_pred.sum(1) + self.smooth)
            self.records.append(ratio.mean())
        else:
            y_pred, y_true = loss_input[0].cpu(), loss_input[1].cpu()
            
            y_pred = y_pred[:, self.target_class, :, :]
            if self.need_exp:
                y_pred = y_pred.exp()
            y_pred = y_pred.view(y_pred.size(0), -1)
            y_true = (y_true == self.target_class)
            y_true = y_true.view(y_pred.size(0), -1).float()
            intersection = (y_pred * y_true).sum(1)
            ratio = 2*(intersection + self.smooth) / (y_true.sum(1) + y_pred.sum(1) + self.smooth)
            self.records.append(ratio.mean())
        
    def output(self):
        dice = sum(self.records) / len(self.records)
        return {'dice': dice}
        
    def reset(self):
        self.records = []