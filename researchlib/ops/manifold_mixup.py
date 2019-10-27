import random
import torch
from torch import nn
import numpy as np
from ..utils import ParameterManager


class _ManifoldMixup(nn.Module):
    block_counter = -1
    prefix_name = 'mmixup_'
    block_id_key = prefix_name + 'block_{}'
    lambda_key = prefix_name + 'lambda'
    index_key = prefix_name + 'index'

    def __init__(self):
        super().__init__()
        self.register()
        self.block_idx = self.block_counter

    def forward(self, x):
        enable = ParameterManager.get_buffer(
            self.block_id_key.format(self.block_idx), clear = False
        )
        if enable:
            lambda_ = ParameterManager.get_buffer(self.lambda_key, clear = False)
            index = ParameterManager.get_buffer(self.index_key, clear = False)
            if not self.training:
                lambda_ = 1.0
                index = list(range(x.size()[0]))
            mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
            return mixed_x
        else:
            return x

    @classmethod
    def reset_counter(cls):
        cls.block_counter = -1

    @classmethod
    def get_y(cls, y):
        index = ParameterManager.get_buffer(cls.index_key, clear = False)
        return y, [i[index] for i in y]

    @classmethod
    def register(cls):
        cls.block_counter = cls.block_counter + 1
        ParameterManager.save_buffer(cls.block_id_key.format(cls.block_counter), False)

    @classmethod
    def setup_batch(cls, alpha, batch_size, fixed_mmixup = None, random_mmixup = None):
        if fixed_mmixup is not None and random_mmixup is not None:
            raise ValueError('Please choose either fixed_mmixup or random_mmixup.')
        else:
            if alpha > 0.:
                lambda_ = np.random.beta(alpha, alpha)
            else:
                lambda_ = 1.
            index = torch.randperm(batch_size).cuda()
            ParameterManager.save_buffer(cls.lambda_key, lambda_)
            ParameterManager.save_buffer(cls.index_key, index)
            if fixed_mmixup is not None:
                block_list = fixed_mmixup
            elif random_mmixup is not None:
                block_list = [random.randint(*random_mmixup)]
            cls.enable_blocks(block_list)
            return lambda_

    @classmethod
    def enable_blocks(cls, block_list):
        if max(block_list) > cls.block_counter:
            raise ValueError(
                'block_idx:{} not exist! deepest block_idx is {} (zero-based)'.format(
                    max(block_list), cls.block_counter
                )
            )
        for block_idx in range(cls.block_counter):
            if block_idx in block_list:
                ParameterManager.save_buffer(cls.block_id_key.format(block_idx), True)
            else:
                ParameterManager.save_buffer(cls.block_id_key.format(block_idx), False)
