import torch
import numpy as np
from ..ops import op
from torch import nn
from .unit.utils import get_act_op, get_norm_op
from ..utils import ParameterManager
from .utils import get_shortcut_op, get_config, BernoulliSkip
from .resblock import _branch_function
from torch import set_grad_enabled



class AdditiveBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, Fm, Gm, *weights):
        '''
            Forward pass for the reversible block computes:
            {x1, x2} = x
            y1 = x1 + Fm(x2)
            y2 = x2 + Gm(y1)
            output = {y1, y2}
            Parameters
            ----------
            ctx : torch.autograd.function.RevNetFunctionBackward
                The backward pass context object
            x : TorchTensor
                Input tensor. Must have channels (2nd dimension) that can be partitioned in two equal partitions
            Fm : nn.Module
                Module to use for computation, must retain dimensions such that Fm(X)=Y, X.shape == Y.shape
            Gm : nn.Module
                Module to use for computation, must retain dimensions such that Gm(X)=Y, X.shape == Y.shape
            *weights : TorchTensor
                weights for Fm and Gm in that order {Fm_w1, ... Fm_wn, Gm_w1, ... Gm_wn}
            Note
            ----
            All tensor/autograd variable input arguments and the output are
            TorchTensors for the scope of this fuction
        '''
        
        # check if possible to partition into two equally sized partitions
        assert(x.shape[1] % 2 == 0)  # nosec

        # store partition size, Fm and Gm functions in context
        ctx.Fm = Fm
        ctx.Gm = Gm

        with torch.no_grad():
            # partition in two equally sized set of channels
            x1, x2 = torch.chunk(x, 2, dim=1)

            # compute outputs
            fmr = Fm.forward(x2)
            y1 = x1 + fmr
            x1.set_()
            del x1
            gmr = Gm.forward(y1)
            y2 = x2 + gmr
            x2.set_()
            del x2
            output = torch.cat([y1, y2], dim=1)
            y1.set_()
            del y1
            y2.set_()
            del y2

        # save the input and output variables (Ming: save only output)
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover

        Fm, Gm = ctx.Fm, ctx.Gm
        # are all variable objects now
        output, = ctx.saved_tensors

        with torch.no_grad():
            y1, y2 = torch.chunk(output, 2, dim=1)

            # partition output gradient also on channels
            assert(grad_output.shape[1] % 2 == 0)  # nosec
            y1_grad, y2_grad = torch.chunk(grad_output, 2, dim=1)

        # Recreate computation graphs for functions Gm and Fm with gradient collecting leaf nodes:
        # z1_stop, x2_stop, GW, FW
        # Also recompute inputs (x1, x2) from outputs (y1, y2)
        with set_grad_enabled(True):
            z1_stop = y1.detach()
            z1_stop.requires_grad = True

            G_z1 = Gm.forward(z1_stop)
            x2 = y2 - G_z1
            x2_stop = x2.detach()
            x2_stop.requires_grad = True

            F_x2 = Fm.forward(x2_stop)
            x1 = y1 - F_x2
            x1_stop = x1.detach()
            x1_stop.requires_grad = True

            # compute outputs building a sub-graph
            y1 = x1_stop + F_x2
            y2 = x2_stop + G_z1

            # calculate the final gradients for the weights and inputs
            dd = torch.autograd.grad(y2, (z1_stop,) + tuple(Gm.parameters()), y2_grad, retain_graph=False)
            z1_grad = dd[0] + y1_grad
            GWgrads = dd[1:]

            dd = torch.autograd.grad(y1, (x1_stop, x2_stop) + tuple(Fm.parameters()), z1_grad, retain_graph=False)

            FWgrads = dd[2:]
            x2_grad = dd[1] + y2_grad
            x1_grad = dd[0]
            grad_input = torch.cat([x1_grad, x2_grad], dim=1)

            y1.detach_()
            y2.detach_()
            del y1, y2

        return (grad_input, None, None) + FWgrads + GWgrads
 

    
class _RevBlock(nn.Module):
    def __init__(self, prefix, _unit, _op, in_dim, out_dim, **kwargs):
        super().__init__()
        parameter_manager = ParameterManager(**kwargs)
        out_dim = int(out_dim / 2)
        config = get_config(prefix, _unit, _op, out_dim, out_dim, parameter_manager)
        preact_bn_shared = parameter_manager.get_param('preact_bn_shared', False) and config.preact and (config.in_dim != config.out_dim or config.do_pool)
        setattr(config, 'preact_bn_shared', preact_bn_shared)
        self.branch_f_op = _branch_function(config, parameter_manager, **kwargs)
        self.branch_g_op = _branch_function(config, parameter_manager, **kwargs)
        
    def forward(self, x):
        args = [x, self.branch_f_op, self.branch_g_op] + [w for w in self.branch_f_op.parameters()] + [w for w in self.branch_g_op.parameters()]
        return AdditiveBlockFunction.apply(*args)