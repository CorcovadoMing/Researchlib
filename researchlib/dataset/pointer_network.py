# coding=utf-8

import numpy as np
import torch
from torch.utils.data import Dataset
from copy import copy


class CHDataset(Dataset):
    """ Dataset for Convex Hull Problem data
    Args:
    filename : the dataset file name
    max_in_seq_len :  maximum input sequence length
    max_out_seq_len : maximum output sequence length
    """

    def __init__(self, filename):
        super().__init__()
        self.max_in_seq_len = 0
        self.max_out_seq_len = 0
        self.START = [0, 0]
        self.END = [0, 0]
        self._load_data(filename)
        print('Max Input Length', self.max_in_seq_len)
        print('Max Output Length', self.max_out_seq_len)

    def _load_data(self, filename):
        with open(filename, 'r') as f:
            data = []
            for line in f:
                inp, outp = line.strip().split('output')
                inp = list(map(float, inp.strip().split(' ')))
                self.max_in_seq_len = max(len(inp), self.max_in_seq_len)
                # Add 1 due to special token
                outp = list(map(int, outp.strip().split(' ')))
                self.max_out_seq_len = max(len(outp), self.max_out_seq_len)
                # Add START token
                outp_in = copy(self.START)
                outp_out = []
                for idx in outp:
                    outp_in += inp[2 * (idx - 1):2 * idx]
                    outp_out += [idx]
                # Add END token
                outp_out += [0]
                data.append((inp, [], outp_in, outp_out, []))

        self.max_in_seq_len = int(self.max_in_seq_len / 2)

        for i in range(len(data)):
            inp, inp_len, outp_in, outp_out, outp_len = data[i]

            # Padding input
            inp_len = len(inp) // 2
            inp = self.START + inp
            inp_len += 1
            # Special START token
            for _ in range(self.max_in_seq_len + 1 - inp_len):
                inp += self.END
            inp = np.array(inp).reshape([-1, 2]).astype('float32')
            inp_len = np.array([inp_len])

            # Padding output
            outp_len = len(outp_out)
            desired_outp_len = self.max_out_seq_len + 1 - outp_len
            for _ in range(desired_outp_len):
                outp_in += self.START
            outp_in = np.array(outp_in).reshape([-1, 2]).astype('float32')
            outp_out = outp_out + [0] * desired_outp_len
            outp_out = np.array(outp_out)
            outp_len = np.array([outp_len])
            data[i] = (inp, inp_len, outp_in, outp_out, outp_len)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inp, inp_len, outp_in, outp_out, outp_len = self.data[index]
        return (inp, inp_len, outp_in), (outp_out, outp_len)
