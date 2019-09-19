from torch import nn
import torch.nn.functional as F
'''
    Weight Standardization
    https://arxiv.org/pdf/1903.10520.pdf
'''


class _WSConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True).mean(dim = 2, keepdim = True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _WSConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups = 1,
        bias = True,
        dilation = 1,
        padding_mode = 'zeros'
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            output_padding = output_padding,
            padding_mode = padding_mode
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True).mean(dim = 2, keepdim = True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _WSConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True).mean(dim = 2, keepdim = True
                                                                ).mean(dim = 3, keepdim = True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _WSConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups = 1,
        bias = True,
        dilation = 1,
        padding_mode = 'zeros'
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            output_padding = output_padding,
            padding_mode = padding_mode
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True).mean(dim = 2, keepdim = True
                                                                ).mean(dim = 3, keepdim = True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _WSConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True).mean(dim = 2, keepdim = True).mean(
            dim = 3, keepdim = True
        ).mean(dim = 4, keepdim = True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class _WSConvTranspose3d(nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        output_padding = 0,
        groups = 1,
        bias = True,
        dilation = 1,
        padding_mode = 'zeros'
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            output_padding = output_padding,
            padding_mode = padding_mode
        )

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim = 1, keepdim = True).mean(dim = 2, keepdim = True).mean(
            dim = 3, keepdim = True
        ).mean(dim = 4, keepdim = True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim = 1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
