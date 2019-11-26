from torch import nn
import torch
from .unit.utils import get_pool_op, get_dim
import torch.nn.functional as F
from ..ops import op


def get_conv_config(**custom_kwargs):
    default = {
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'dilation': 1,
        'groups': 1,
        'bias': False
    }
    default.update(**custom_kwargs)
    return default


class padding_shortcut(nn.Module):
    def __init__(self, _op, in_dim, out_dim, norm_op, shortcut_norm, do_pool, pool_factor, blur, transpose, stride):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        pool_op = get_pool_op('avg', get_dim(_op), pool_factor) if do_pool else None    
        self.pool_op = nn.Sequential(*list(filter(None, [pool_op])))

    def forward(self, x):
        x = self.pool_op(x)
        if self.in_dim >= self.out_dim:
            return x[:, :self.out_dim]
        else:
            return torch.cat(
                (
                    x,
                    torch.zeros((x.size(0), self.out_dim - self.in_dim, *x.shape[2:]), device = x.device).to(x.dtype)
                ), 1)
        
        
class projection_shortcut(nn.Module):
    def __init__(self, _op, in_dim, out_dim, norm_op, shortcut_norm, do_pool, pool_factor, blur, transpose, stride):
        super().__init__()
        if in_dim != out_dim or do_pool:
            shortcut_kernel_size = 2 if transpose and do_pool else 1
            shortcut_stride = 1 if blur else stride
            reduction_op = [
                _op(
                    in_dim,
                    out_dim,
                    kernel_size = shortcut_kernel_size,
                    stride = shortcut_stride,
                    bias = False
                ),
                norm_op if shortcut_norm else None
            ]
            if blur:
                reduction_op.append(
                    op.Downsample(channels = self.out_dim, filt_size = 3, stride = stride)
                )
        else:
            reduction_op = [None]
        self.op = nn.Sequential(*list(filter(None, reduction_op)))
    
    def forward(self, x):
        return self.op(x)
    
    

class SE_Attention(nn.Module):
    def __init__(self, dim, dim_type, ratio = 16):
        super().__init__()
        self.fn = nn.Sequential(
            op.__dict__['AdaptiveAvgPool' + dim_type](1),
            op.__dict__['Conv' + dim_type](dim, dim // ratio, kernel_size = 1),
            nn.ReLU(inplace = True),
            op.__dict__['Conv' + dim_type](dim // ratio, dim, kernel_size = 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fn(x)


def _channel_pool(x):
    return torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim = 1)


class CBAM_Attention(nn.Module):
    def __init__(self, dim, dim_type, ratio = 16):
        super().__init__()

        # Channel
        self.max_pool = op.__dict__['AdaptiveMaxPool' + dim_type](1)
        self.avg_pool = op.__dict__['AdaptiveAvgPool' + dim_type](1)

        self.cfn = nn.Sequential(
            op.__dict__['Conv' + dim_type](dim, dim // ratio, kernel_size = 1),
            nn.ReLU(inplace = True),
            op.__dict__['Conv' + dim_type](dim // ratio, dim, kernel_size = 1)
        )

        # Spatial
        self.sfn = nn.Sequential(
            op.__dict__['Conv' + dim_type](2, 1, 3, 1, 1),
            op.__dict__['BatchNorm' + dim_type](1), nn.Sigmoid()
        )

    def forward(self, x):
        channel_attention = x * torch.sigmoid(
            self.cfn(self.max_pool(x)) + self.cfn(self.avg_pool(x))
        )
        spatial_attention = channel_attention * self.sfn(_channel_pool(channel_attention))
        return spatial_attention


