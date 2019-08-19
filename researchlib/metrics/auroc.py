from .matrix import Matrix
import numpy as np
from sklearn.metrics import roc_auc_score


class AUROC(Matrix):

    def __init__(self):
        super().__init__()
        self.pred = None
        self.true = None

    def forward(self, loss_input):
        if len(loss_input) == 5:
            # mixup (INCOMPLETE!!)
            pass
        else:
            y_pred, y_true = loss_input[0], loss_input[1]
            y_pred, y_true = y_pred.detach(), y_true.detach()
            y_pred = y_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()
            if self.pred is None:
                self.pred = y_pred
            else:
                self.pred = np.concatenate((self.pred, y_pred))
            if self.true is None:
                self.true = y_true
            else:
                self.true = np.concatenate((self.true, y_true))

    def output(self):
        auc = roc_auc_score(self.true, self.pred)
        return {'auroc': auc}

    def reset(self):
        self.pred = None
        self.true = None
