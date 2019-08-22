from torch.autograd import Function
import torch
from torch import nn
import torch.nn.functional as F


class _blur_function_backward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        grad_input = F.conv2d(grad_output, kernel_flip, padding=1, groups=grad_output.size(1))
        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = F.conv2d(gradgrad_output, kernel, padding=1, groups=gradgrad_output.size(1))
        return grad_input, None, None


class _blur_function(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)
        output = F.conv2d(input, kernel, padding=1, groups=input.size(1))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors
        grad_input = _blur_function_backward.apply(grad_output, kernel, kernel_flip)
        return grad_input, None, None

    
class _Blur2d(nn.Module):
    def __init__(self, channel):
        super().__init__()
        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])
        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return _blur_function.apply(input, self.weight, self.weight_flip)