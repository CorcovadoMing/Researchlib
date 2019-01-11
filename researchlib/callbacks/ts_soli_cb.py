from .callback import Callback
from ..io import *
import numpy as np

class SoliAcc(Callback):
    def __init__(self, total_class=11, masked_class=10, name=None):
        super().__init__()
        self.total_class = total_class
        self.masked_class = masked_class
        if name:
            self.exp_name = name + '.h5'
        else:
            self.exp_name = 'tmp.h5'
        self.best_acc = -1
    
    def on_validation_end(self, **kwargs):
        total = [0]*self.total_class
        acc = [0]*self.total_class
            
        for i, (x, y) in enumerate(kwargs['test_loader']):
            t = kwargs['model'](x.cuda())
            t = t.squeeze().argmax(-1)
            tx = t.cpu().numpy().astype(int)
            ty = y.squeeze().numpy().astype(np.int)
            
            for i in range(len(tx)):
                m = set(list(tx[i]))
                m.remove(0)
                t = set(list(ty[i]))
                t.remove(0)
                for j in t:
                    total[int(j)] += 1
                    if j in m:
                        acc[int(j)] += 1
        rate = sum(acc[:self.masked_class+1]) / (float(sum(total[:self.masked_class+1]))+1e-5)
        
        print(total, acc, rate)
        
        if rate >= self.best_acc:
            self.best_acc = rate
            save_model(kwargs['model'], self.exp_name)
            print('Saving model to ' + self.exp_name)
        