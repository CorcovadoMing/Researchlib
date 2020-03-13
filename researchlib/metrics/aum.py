import torch


class AUM:
    '''
        Area Under Margin
        The logit of assigned class - max of rest logit
    '''
    def __call__(self, x, y):
        with torch.no_grad():
            assign = x[torch.arange(len(y)), y]
            masked = x.clone()
            masked[torch.arange(len(y)), y] = 0
            return assign - masked.max(-1)[0]