import torch
import torch.nn as nn
from torch.autograd import Function



def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)
    
    
def _threshold_and_support(input, dim=0):
    """Sparsemax building block: compute the threshold

    Args:
        input: any dimension
        dim: dimension along which to apply the sparsemax

    Returns:
        the threshold value
    """

    input_srt, _ = torch.sort(input, descending=True, dim=dim)
    input_cumsum = input_srt.cumsum(dim) - 1
    rhos = _make_ix_like(input, dim)
    support = rhos * input_srt > input_cumsum

    support_size = support.sum(dim=dim).unsqueeze(dim)
    tau = input_cumsum.gather(dim, support_size - 1)
    tau /= support_size.to(input.dtype)
    return tau, support_size



class SparsemaxLossFunction(Function):

    @staticmethod
    def forward(ctx, input, target):
        """
        input (FloatTensor): ``(n, num_classes)``.
        target (LongTensor): ``(n,)``, the indices of the target classes
        """
        input_batch, classes = input.size()
        target_batch = target.size(0)
        aeq(input_batch, target_batch)

        z_k = input.gather(1, target.unsqueeze(1)).squeeze()
        tau_z, support_size = _threshold_and_support(input, dim=1)
        support = input > tau_z
        x = torch.where(
            support, input**2 - tau_z**2,
            torch.tensor(0.0, device=input.device)
        ).sum(dim=1)
        ctx.save_for_backward(input, target, tau_z)
        # clamping necessary because of numerical errors: loss should be lower
        # bounded by zero, but negative values near zero are possible without
        # the clamp
        return torch.clamp(x / 2 - z_k + 0.5, min=0.0)

    @staticmethod
    def backward(ctx, grad_output):
        input, target, tau_z = ctx.saved_tensors
        sparsemax_out = torch.clamp(input - tau_z, min=0)
        delta = torch.zeros_like(sparsemax_out)
        delta.scatter_(1, target.unsqueeze(1), 1)
        return sparsemax_out - delta, None


sparsemax_loss = SparsemaxLossFunction.apply


class SparsemaxLoss(nn.Module):
    """
    An implementation of sparsemax loss, first proposed in
    :cite:`DBLP:journals/corr/MartinsA16`. If using
    a sparse output layer, it is not possible to use negative log likelihood
    because the loss is infinite in the case the target is assigned zero
    probability. Inputs to SparsemaxLoss are arbitrary dense real-valued
    vectors (like in nn.CrossEntropyLoss), not probability vectors (like in
    nn.NLLLoss).
    """

    def __init__(self, weight=None, ignore_index=-100,
                 reduction='elementwise_mean'):
        assert reduction in ['elementwise_mean', 'sum', 'none']
        self.reduction = reduction
        self.weight = weight
        self.ignore_index = ignore_index
        super().__init__()

    def forward(self, input, target):
        loss = sparsemax_loss(input, target)
        if self.ignore_index >= 0:
            ignored_positions = target == self.ignore_index
            size = float((target.size(0) - ignored_positions.sum()).item())
            loss.masked_fill_(ignored_positions, 0.0)
        else:
            size = float(target.size(0))
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'elementwise_mean':
            loss = loss.sum() / size
        return loss