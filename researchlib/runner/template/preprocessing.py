import torch


class NumpyPreprocessing(object):
    def __init__(self):
        pass

    def _forward(self, x, y):
        x, y = [i.numpy().copy() for i in x], [i.numpy().copy() for i in y]
        x, y = self.forward_batch(x, y)
        try:
            x, y = [torch.from_numpy(i)
                    for i in x], [torch.from_numpy(i) for i in y]
        except:
            x, y = [torch.from_numpy(i.copy())
                    for i in x], [torch.from_numpy(i.copy()) for i in y]
        return x, y

    def forward_batch(self, x, y):
        batch_size = len(x[0])
        for cur_batch in range(batch_size):
            single_x = [i[cur_batch] for i in x]
            single_y = [i[cur_batch] for i in y]
            single_x, single_y = self.forward_single(single_x, single_y)
            for i in range(len(x)):
                x[i][cur_batch] = single_x[i]
            for i in range(len(y)):
                y[i][cur_batch] = single_y[i]
        return x, y

    def forward_single(self, x, y):
        raise ('Not implemented')


class TorchPreprocessing(object):
    def __init__(self):
        pass

    def _forward(self, x, y):
        x = [i.clone() for i in x]
        y = [i.clone() for i in y]
        x, y = self.forward_batch(x, y)
        return x, y

    def forward_batch(self, x, y):
        batch_size = len(x[0])
        for cur_batch in range(batch_size):
            single_x = [i[cur_batch] for i in x]
            single_y = [i[cur_batch] for i in y]
            single_x, single_y = self.forward_single(single_x, single_y)
            for i in range(len(x)):
                x[i][cur_batch] = single_x[i]
            for i in range(len(y)):
                y[i][cur_batch] = single_y[i]
        return x, y

    def forward_single(self, x, y):
        raise ('Not implemented')
