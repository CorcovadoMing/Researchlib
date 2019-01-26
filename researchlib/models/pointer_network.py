# coding=utf-8

import torch
from torch import nn
from torch.autograd import Variable

from ..layers import *

class PointerNetRNNDecoder(RNNDecoderBase):
    def __init__(self, rnn_type, bidirectional, num_layers,
        input_size, hidden_size, dropout):
        super().__init__(rnn_type, bidirectional, num_layers, input_size, hidden_size, dropout)
        self.attention = Attention("dot", hidden_size)

    def forward(self, tgt, memory_bank, hidden, memory_lengths=None):
        # RNN
        rnn_output, hidden_final = self.rnn(tgt, hidden)
        # Attention
        memory_bank = memory_bank.transpose(0, 1)
        rnn_output = rnn_output.transpose(0, 1)
        attn_h, align_score = self.attention(memory_bank, rnn_output, memory_lengths)
        return align_score

class PointerNet(nn.Module):
    """ Pointer network
    Args:
    rnn_type (str) : rnn cell type
    bidirectional : whether rnn is bidirectional
    num_layers : number of layers of stacked rnn
    encoder_input_size : input size of encoder
    rnn_hidden_size : rnn hidden dimension size
    dropout : dropout rate
    """
    def __init__(self, rnn_type, bidirectional, num_layers, encoder_input_size, rnn_hidden_size, dropout):
        super().__init__()
        self.encoder = RNNEncoder(rnn_type, bidirectional,
          num_layers, encoder_input_size, rnn_hidden_size, dropout)
        self.decoder = PointerNetRNNDecoder(rnn_type, bidirectional,
          num_layers, encoder_input_size, rnn_hidden_size, dropout)

    def forward(self, inp, inp_len, outp):
        inp = inp.transpose(0, 1)
        outp = outp.transpose(0, 1)
        memory_bank, hidden_final = self.encoder(inp, inp_len)
        align_score = self.decoder(outp, memory_bank, hidden_final, inp_len)
        return align_score