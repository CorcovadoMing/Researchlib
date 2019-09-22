import torch
import random
import matplotlib.pyplot as plt


class NumpyAugmentation(object):
    def __init__(self):
        self.prob = None
        self.mag = None
        self._debug_flag = False

    def _forward(self, x, y, prob, mag):
        self.cur_prob = self.prob or prob
        self.cur_mag = self.mag or mag

        if self._debug_flag:
            print('Probability:', self.cur_prob, 'Magnitude:', self.cur_mag)

        if random.random() < prob:
            x, y = [i.numpy().copy() for i in x], [i.numpy().copy() for i in y]
            x, y = self.forward_batch(x, y, mag)
            try:
                x, y = [torch.from_numpy(i) for i in x], [torch.from_numpy(i) for i in y]
            except:
                x, y = [torch.from_numpy(i.copy())
                        for i in x], [torch.from_numpy(i.copy()) for i in y]
        return x, y

    def forward_batch(self, x, y, mag):
        batch_size = len(x[0])
        for cur_batch in range(batch_size):
            single_x = [i[cur_batch] for i in x]
            single_y = [i[cur_batch] for i in y]
            single_x, single_y = self.forward_single(single_x, single_y, mag)
            for i in range(len(x)):
                x[i][cur_batch] = single_x[i]
            for i in range(len(y)):
                y[i][cur_batch] = single_y[i]
        return x, y

    def forward_single(self, x, y, mag):
        raise ('Not implemented')


class TorchAugmentation(object):
    def __init__(self):
        self.prob = None
        self.mag = None
        self._debug_flag = False

    def _forward(self, x, y, prob, mag):
        self.cur_prob = self.prob or prob
        self.cur_mag = self.mag or mag

        if self._debug_flag:
            print('Probability:', self.cur_prob, 'Magnitude:', self.cur_mag)

        if random.random() < prob:
            x = [i.clone() for i in x]
            y = [i.clone() for i in y]
            x, y = self.forward_batch(x, y, mag)
        return x, y

    def forward_batch(self, x, y, mag):
        batch_size = len(x[0])
        for cur_batch in range(batch_size):
            single_x = [i[cur_batch] for i in x]
            single_y = [i[cur_batch] for i in y]
            single_x, single_y = self.forward_single(single_x, single_y, mag)
            for i in range(len(x)):
                x[i][cur_batch] = single_x[i]
            for i in range(len(y)):
                y[i][cur_batch] = single_y[i]
        return x, y

    def forward_single(self, x, y, mag):
        raise ('Not implemented')
