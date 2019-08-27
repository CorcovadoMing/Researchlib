from .matrix import *
import torch
import numpy as np


class MAPE(Matrix):
    """ The mean absolute percentage error (MAPE) is a statistical measure of how accurate a forecast system is. It measures this accuracy as a percentage, and can be calculated as the average absolute percent error for each time period minus actual values divided by actual values. Where At is the actual value and Ft is the forecast value, this is given by: $formula := mean_t(|(At-Ft)/At|)$
    The mean absolute percentage error (MAPE) is the most common measure used to forecast error, and works best if there are no extremes to the data (and no zeros).
    """

    def __init__(self):
        super().__init__()
        self.total = 0
        self.value = 0

    def forward(self, loss_input):
        y_pred, y_true = loss_input[0].cpu().detach().numpy(
        ), loss_input[1].cpu().detach().numpy()
        mask = (y_true != 0.0)
        self.value += (np.abs(y_true[mask] - y_pred[mask]) /
                       y_true[mask]).mean()
        self.total += 1

    def output(self):
        mape = (self.value / float(self.total))
        return {'mape': mape}

    def reset(self):
        self.total = 0
        self.value = 0
