"""aLSTM helpers

Support functions for the aLSTM.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


class VariationalDropout(object):

    """Multi-layer Variation Dropout.

    Module for applying Variational Dropout across time in a multi-layer aLSTM.
    Pre-constructs dropout masks for each layer, so that when called upon the
    same mask is applied across time.

    ..warning ::
        The masks are built in the constructor, so each forward
        pass should use a separate instance of ::class::`VariationalDropout`.

    Args:
        tensor (Variable): example data source to allocate masks on right device.
        dropouts (list, float): list of drop probabilities.
        sizes (list): list of mask sizes.
    """

    def __init__(self, tensor, dropouts, sizes):
        if isinstance(dropouts, (float, int)):
            dropouts = [dropouts] * len(sizes)
        masks = []
        for p, size in zip(dropouts, sizes):
            mask = tensor.new(*size).bernoulli_(1 - p) / (1 - p)
            masks.append(mask)

        self.dropouts = dropouts
        self.masks = masks

    def __call__(self, input, layer_idx):
        return input * Variable(self.masks[layer_idx], requires_grad=False)


class Project(nn.Module):

    """Projection into sub-policies

    Projection from a latent variable into sub-policies.
    """

    def __init__(self, ninp, nout):
        super(Project, self).__init__()
        self.ninp = ninp
        self.nout = nout
        self.linear = nn.Linear(ninp, nout, bias=False)

    def forward(self, input):
        return torch.tanh(self.linear(input))


def chunk(tensor, sizes, dim=0):
    """Splits the tensor according to chunks of sizes along dim.

        Arguments:
            tensor (Tensor): tensor to split.
            sizes (list): sizes of chunks
            dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    if tensor.size(dim) != sum(sizes):
        raise ValueError(
            "Chunk sizes ({}) do not match tensor size ({}) in dim {}".format(
                sum(sizes), tensor.size(dim), dim))

    nsizes = len(sizes)
    sizes = [0] + sizes
    return [tensor.narrow(dim, sizes[i], sizes[i+1]) for i in range(nsizes)]


def get_sizes(input_size, hidden_size, output_size, layer_idx, nlayers):
    """Get the input and hidden size for a layer"""
    if nlayers == 1:
        return input_size, output_size
    if layer_idx == 0:
        return input_size, hidden_size
    if layer_idx == nlayers - 1:
        return hidden_size, output_size
    return hidden_size, hidden_size


def init_hidden(data_source, bsz, asz, osz, hsz, nlayers):
    """Utility for initializing a stack of hidden state tuples"""

    def hidden(out):
        return Variable(data_source.new(bsz, out).zero_())

    ah = [(hidden(asz), hidden(asz)) for _ in range(nlayers)]
    fh = [(hidden(hsz if l != nlayers - 1 else osz),
           hidden(hsz if l != nlayers - 1 else osz))
          for l in range(nlayers)]
    return ah, fh


def convert(hiddens, fmt):
    """Convert the inner format of a nested list of lists or tuples"""
    return tuple([fmt(h) for h in h_list] for h_list in hiddens)
