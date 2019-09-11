import re
import torch
import copy
from torch import nn
from ...layers import layer
from ...utils import ParameterManager
from .attention import _SE_Attention, _CBAM_Attention
from .shortcut import _padding_shortcut


class _Block(nn.Module):

    def __init__(self, op, in_dim, out_dim, do_pool, do_norm, preact, **kwargs):
        super().__init__()
        self.op = op
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.do_pool = do_pool
        self.do_norm = do_norm
        self.preact = preact
        self.parameter_manager = ParameterManager(**kwargs)

        # Customize
        self.__postinit__()

    def __postinit__(self):
        pass

    def _get_conv_kwargs(self):
        kernel_size = self._get_param('kernel_size', 3)
        stride = self._get_param('stride', 1)
        padding = self._get_param('padding', 1)
        dilation = self._get_param('dilation', 1)
        groups = self._get_param('groups', 1)
        bias = self._get_param('bias', False)
        return {
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'bias': bias
        }

    def _get_custom_kwargs(self, custom_kwargs={}):
        _new = copy.deepcopy(self.parameter_manager.kwargs)
        for key, value in custom_kwargs.items():
            _new[key] = value
        return _new

    def _get_param(self, key, init_value=None, required=False):
        return self.parameter_manager.get_param(key, init_value, required)

    def _get_dim_type(self):
        match = re.search('\dd', str(self.op))
        dim_str = match.group(0)
        return dim_str

    def _is_transpose(self):
        match = re.search('Transpose', str(self.op))
        return True if match is not None else False

    def _get_activator_layer(self, activator_type):
        if activator_type not in [
                'ReLU', 'ELU', 'PReLU', 'LeakyReLU', 'SELU', 'Swish', 'GeLU',
                'CELU', 'DropReLU', 'Mish'
        ]:
            raise ValueError('Unknown activator type')
        
        # Inplace
        if activator_type in ['ReLU']:
            act_kwargs = {'inplace': True}
        else:
            act_kwargs = {}
        return layer.__dict__[activator_type](**act_kwargs)

    def _get_norm_layer(self, norm_type, dim=None):
        if norm_type not in ['BatchNorm', 'InstanceNorm', 'GroupNorm']:
            raise ValueError('Unknown norm type')

        if dim is None:
            if self.preact:
                dim = [self.in_dim]
            else:
                dim = [self.out_dim]
        else:
            dim = [dim]

        if norm_type is not 'GroupNorm':
            dim_str = self._get_dim_type()
        else:
            dim_str = ''
            group_num = self._get_param('groupnorm_group', 4)
            dim.insert(0, group_num)
        norm_op_str = norm_type + dim_str
        norm_op = layer.__dict__[norm_op_str]

        return norm_op(*dim)

    def _get_pool_layer(self, pool_type, pool_factor, dim):
        if pool_type not in ['MaxPool', 'AvgPool', 'Combined', 'Upsample']:
            raise ValueError('Unknown pool type')

        dim_str = self._get_dim_type()
        if pool_type is 'Upsample':
            pool_op = layer.__dict__[pool_type]
            return pool_op(scale_factor=pool_factor)
        
        elif pool_type is not 'Combined':
            pool_op_str = pool_type + dim_str
            pool_op = layer.__dict__[pool_op_str]
            
            blur = self._get_param('blur', False)
            if pool_type == 'MaxPool' and blur:
                pool_layer = nn.Sequential(
                    pool_op(pool_factor, stride=1),
                    layer.Downsample(channels=dim, filt_size=3, stride=pool_factor)
                )
            elif pool_type == 'AvgPool' and blur:
                pool_layer = nn.Sequential(
                    layer.Downsample(channels=dim, filt_size=3, stride=pool_factor)
                )
            else:
                pool_layer = pool_op(pool_factor)
            
            return pool_layer
        
        else:
            max_pool_op = layer.__dict__['MaxPool' + dim_str](pool_factor)
            avg_pool_op = layer.__dict__['AvgPool' + dim_str](pool_factor)
            conv_pool_op = nn.Sequential(layer.__dict__['Conv' + dim_str](
                self.out_dim, self.out_dim, 4, 2, 1),
                                         nn.LeakyReLU(0.5))  # Special case
            reduction_op = layer.__dict__['Conv' + dim_str](self.out_dim * 3,
                                                            self.out_dim, 1)
            return _Combined([max_pool_op, avg_pool_op, conv_pool_op],
                             reduction_op, self.preact)

    def _get_attention_branch(self, ratio=16, dim=None):
        if dim is None:
            dim = self.out_dim
        attention_type = self._get_param('branch_attention', required=True)
        if attention_type == 'SE':
            return _SE_Attention(dim, self._get_dim_type(), ratio)
        elif attention_type == 'CBAM':
            return _CBAM_Attention(dim, self._get_dim_type(), ratio)
        else:
            raise ValueError('Unknown branch attention type')

    def _get_shake_drop_branch(self):
        id = self._get_param('id', required=True)
        total_blocks = self._get_param('total_blocks', required=True)
        alpha_range = self._get_param('alpha_range', init_value=[-1, 1])
        beta_range = self._get_param('beta_range', init_value=[0, 1])
        shakedrop_prob = self._get_param('shakedrop_prob', init_value=0.5)
        mode_mapping = {'batch': 0, 'sample': 1, 'channel': 2, 'pixel': 3}
        mode = self._get_param('shakedrop_mode', 'pixel')
        mode = mode_mapping[mode]
        return layer.ShakeDrop(
            id,
            total_blocks,
            alpha_range=alpha_range,
            beta_range=beta_range,
            shakedrop_prob=shakedrop_prob,
            mode=mode)
    
    def _get_shortcut(self):
        blur = self._get_param('blur', False) and self.do_pool
        is_transpose = self._is_transpose()
        stride = self._get_param('pool_factor', 2) if self.do_pool else 1
        shortcut_type = self._get_param('shortcut', 'projection')
        if shortcut_type not in ['projection', 'padding']:
            raise ('Shortcut type is not supported')
        if shortcut_type == 'projection':
            shortcut_kernel_size = 2 if is_transpose and self.do_pool else 1
            shortcut_stride = 1 if blur else stride
            if self.in_dim != self.out_dim or self.do_pool:
                reduction_op = [self.op(
                    self.in_dim,
                    self.out_dim,
                    kernel_size=shortcut_kernel_size,
                    stride=shortcut_stride)]
                if blur:
                    reduction_op.append(layer.Downsample(channels=self.out_dim, filt_size=3, stride=stride))
            else:
                reduction_op = [None]
        elif shortcut_type == 'padding':
            pool_type = self._get_param('pool_type', 'AvgPool')  # As paper's design
            pool_factor = self._get_param('pool_factor', 2)
            pool_layer = self._get_pool_layer(pool_type, pool_factor, self.in_dim) if self.do_pool else None
            reduction_op = [_padding_shortcut(self.in_dim, self.out_dim, pool_layer)]
        return nn.Sequential(*list(filter(None, reduction_op)))
    

    def forward(self, x):
        pass


# ================================================================================================


class _Combined(nn.Module):

    def __init__(self, fn_list, reduction_op, preact):
        super().__init__()
        self.fn_list = nn.ModuleList(fn_list)
        if preact:
            self.reduction_op = reduction_op
        else:
            self.reduction_op = nn.Sequential(reduction_op, nn.LeakyReLU(0.5))

    def forward(self, x):
        out = [f(x) for f in self.fn_list]
        out = torch.cat(out, dim=1)
        return self.reduction_op(out)
