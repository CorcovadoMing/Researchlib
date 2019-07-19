import torch.nn.utils.spectral_norm as sn
import torch
from torch import nn
import torch.nn.functional as F
from ..layers import layer
import functools
from torch.autograd import Function
from torch.nn import init
from math import sqrt
import random
import numpy as np

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, fused=True):
        super().__init__()

        self.conv = nn.Sequential(
            *[
                EqualConv2d(6, 32, 1),
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(128, 256, 3, 1, downsample=True),  # 32
                ConvBlock(256, 256, 3, 1, downsample=True),  # 16
                ConvBlock(256, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, 4, 0),
                layer.Reshape((-1, 512))
            ]
        )
        self.linear = EqualLinear(512, 1)
        
    def forward(self, input):
        bs, ch, w, h = input.shape
        input = input.reshape(bs//2, ch*2, w, h)
        out = self.conv(input)
        return self.linear(out)
        
class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
        id=0
    ):
        super().__init__()
        
        self.id = id
        if self.id == 0:
            initial = True

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel, True))
        self.adain1 = AdaIn2d(out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel, True))
        self.adain2 = AdaIn2d(out_channel)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        input, style, cross_point = x
        if cross_point == -1:
            _style = style
        elif self.id < cross_point:
            _style = style[:int(style.size(0)/2)]
        elif self.id >= cross_point:
            _style = style[int(style.size(0)/2):]
        out = self.conv1(input)
        out = self.noise1(out)
        out = self.lrelu1(out)
        out = self.adain1(out, _style)

        out = self.conv2(out)
        out = self.noise2(out)
        out = self.lrelu2(out)
        out = self.adain2(out, _style)
        return (out, style, cross_point)


class Generator(nn.Module):
    def __init__(self, mixing_regularization=False, truncate=1):
        super().__init__()
        
        if mixing_regularization:
            self.multi_sample = 2
        else:
            self.multi_sample = 1
        
        self.truncate = truncate
        
        self.projection = nn.Sequential(*[
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2),
            EqualLinear(512, 512),
            nn.LeakyReLU(0.2)
        ])
        
        blocks = []
        resolution = 4 # constant
        in_dim, out_dim = 512, 512
        for i in range(6): # 2^(8-2) = 128
            if i > 0:
                resolution *= 2
            in_dim = out_dim
            if i != 0 and i % 2 == 0:
                out_dim /= 2
            if resolution == 128:
                out_dim /= 2
                fused = True
            else:
                fused = False
            print(in_dim, out_dim, resolution)
            blocks.append(StyledConvBlock(int(in_dim), int(out_dim), 3, 1, upsample=True, fused=fused, id=i))
        
        self.mixing_max_range = 6+1
        
#         self.main = nn.Sequential(*[
#             StyledConvBlock(512, 512, 3, 1, initial=True, style=1),  # 4
#             StyledConvBlock(512, 512, 3, 1, upsample=True, style=1),  # 8
#             StyledConvBlock(512, 256, 3, 1, upsample=True, style=1),  # 16
#             StyledConvBlock(256, 256, 3, 1, upsample=True, style=2),  # 32
#             StyledConvBlock(256, 128, 3, 1, upsample=True, style=2),  # 64
#             StyledConvBlock(128, 64, 3, 1, upsample=True, fused=True, style=2)  # 128
#         ])
        self.main = nn.Sequential(*blocks)
        
        self.out = EqualConv2d(64, 3, 1)
        self.tanh = nn.Tanh()

    def _truncat_trick(self, w, truncate):
        w_bar = torch.mean(w, dim=-1).unsqueeze(-1)
        return w_bar + truncate * (w_bar - w)
        
    def forward(self, input):
        if self.multi_sample > 1:
            w1, w2 = input[:int(input.size(0)/self.multi_sample)], input[int(input.size(0)/self.multi_sample):]
            w1, w2 = self.projection(w1), self.projection(w2)
            w1 = self._truncat_trick(w1, self.truncate)
            w2 = self._truncat_trick(w2, self.truncate)
            out = torch.cat([w1, w2], dim=0)
            cross_point = np.random.randint(0, self.mixing_max_range)
            out, _, _ = self.main((w1, out, cross_point))
        else:
            out = self.projection(input)
            out = self._truncat_trick(out, self.truncate)
            out, _, _ = self.main((out, out, -1))
        return self.tanh(self.out(out))
        

# ============================================================================


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, intermediate_features=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.intermediate_features = intermediate_features

        self.T = nn.Parameter(
            torch.Tensor(in_features, out_features - in_features, intermediate_features)
        )

    def forward(self, x):
        """Computes the output of the Minibatch Discrimination Layer

        Args:
            (N, infeatures)

        Returns:
            (N, outfeatures)
        """
        M = torch.mm(x, self.T.view(self.in_features, -1))
        M = M.view(-1, self.out_features-self.in_features, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        # Broadcasting reduces the matrix subtraction to the form desired in the paper
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        return torch.cat([x, out], 1)

class SelfModNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_features, affine=False)
        hidden_features = num_features // 4
        self.gamma_f1 = sn(nn.Linear(64, hidden_features))
        self.beta_f1 = sn(nn.Linear(64, hidden_features))
        self.gamma_f2 = sn(nn.Linear(hidden_features, num_features))
        self.beta_f2 = sn(nn.Linear(hidden_features, num_features))

    def forward(self, x, y):
        out = self.norm(x)
        gamma = self.gamma_f2(F.leaky_relu(self.gamma_f1(y), 0.2, inplace=True))
        beta = self.beta_f2(F.leaky_relu(self.beta_f1(y), 0.2, inplace=True))
        return (1+gamma).view(out.size(0), -1, 1, 1) * out + beta.view(out.size(0), -1, 1, 1)


class AdaIn2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_f = sn(nn.Linear(512, num_features))
        self.beta_f = sn(nn.Linear(512, num_features))

    def forward(self, x, y):
        out = self.norm(x)
        gamma = self.gamma_f(y)
        beta = self.beta_f(y)
        return gamma.view(out.size(0), -1, 1, 1) * out + beta.view(out.size(0), -1, 1, 1)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # Channel multiplier
        self.ch = ch
        self.theta = sn(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
        self.phi = sn(nn.Conv2d(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False))
        self.g = sn(nn.Conv2d(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False))
        self.o = sn(nn.Conv2d(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False))
        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        
    def forward(self, input):
        if type(input) == tuple:
            x, y = input
        else:
            x = input
        # Apply convs
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])    
        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        # Matmul and softmax to get attention maps
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        if type(input) == tuple:
            return (self.gamma * o), y
        else:
            return self.gamma * o


class NoiseInjection(nn.Module):
    def __init__(self, channel, enable):
        super().__init__()
        self.enable = enable
        if enable:
            self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
        
    def forward(self, input):
        if self.enable:
            noise = torch.empty_like(input).to(input.device).normal_()
            return input + self.weight * noise
        else:
            return input


class block_g(nn.Module):
    def __init__(self, in_dim, out_dim, id=0, do_upsample=True, inject_noise=False):
        super().__init__()
        self.id = id
        self.inject_noise = inject_noise
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        hidden_dim = in_dim // 4
        self.conv1 = sn(torch.nn.Conv2d(in_dim, hidden_dim, 1, bias=False))
        self.conv2 = sn(torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
        self.conv3 = sn(torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
        self.conv4 = sn(torch.nn.Conv2d(hidden_dim, out_dim, 1, bias=False))
        
        self.n1 = NoiseInjection(in_dim, inject_noise)
        self.n2 = NoiseInjection(hidden_dim, inject_noise)
        self.n3 = NoiseInjection(hidden_dim, inject_noise)
        self.n4 = NoiseInjection(hidden_dim, inject_noise)
        
#         self.bn1 = SelfModNorm2d(in_dim)
#         self.bn2 = SelfModNorm2d(hidden_dim)
#         self.bn3 = SelfModNorm2d(hidden_dim)
#         self.bn4 = SelfModNorm2d(hidden_dim)
        self.bn1 = AdaIn2d(in_dim)
        self.bn2 = AdaIn2d(hidden_dim)
        self.bn3 = AdaIn2d(hidden_dim)
        self.bn4 = AdaIn2d(hidden_dim)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.do_upsample = do_upsample
        
    def forward(self, input):
        x, y = input
#         _y = y[:, self.id*64:(self.id+1)*64]
        _y = y
        
        h = self.conv1(self.act(self.bn1(self.n1(x), _y)))
        h = self.act(self.bn2(self.n2(h), _y))
        
        if self.in_dim != self.out_dim:
            # Drop channels
            x = x[:, :self.out_dim]
        
        if self.do_upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        
        h = self.conv2(h)
        h = self.conv3(self.act(self.bn3(self.n3(h), _y)))
        h = self.conv4(self.act(self.bn4(self.n4(h), _y)))
        return (h+x, y)
        
        
class ToRGB(nn.Module):
    def __init__(self, in_dim, id=0):
        super().__init__()
        self.f = block_g(in_dim, 3, id, do_upsample=False)
        self.out = nn.Tanh()
        
    def forward(self, input):
        x, y = self.f(input)
        return (self.out(x), y)

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out

class AutoGAN_G(torch.nn.Module):
    def __init__(self, img_size, base_hidden=16, attention=64, const_input=False, inject_noise=False):
        super().__init__()
        
        self.inject_noise = inject_noise
        
        self.attention = False
        if attention is not None:
            self.attention = True
            self.attention_resolution = attention
        
        _block = block_g
        
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        kt = img_size // 4
        mult = img_size // 4
        img_new_size = 4
        
        i = 0
        while img_new_size != img_size:
            print(img_new_size, mult*base_hidden, base_hidden*(mult//2))
            if self.attention and img_new_size == self.attention_resolution:
                main.add_module('Att [%d]' % i, SelfAttention(base_hidden*mult))
            main.add_module('Middle-block [%d]' % i, _block(base_hidden*mult, base_hidden*(mult//2), id=i, inject_noise=self.inject_noise))
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1
            img_new_size *= 2

        print(i)
        
        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
#         if sa:
#             main.add_module('End-Att', SelfAttention(base_hidden))
        main.add_module('End-Out', ToRGB(base_hidden, id=i))
        # Size = n_colors x image_size x image_size
        self.main = main
        
        self.projection = nn.Sequential(*[
            sn(nn.Linear(512, 512, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            sn(nn.Linear(512, 512, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            sn(nn.Linear(512, 512, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            sn(nn.Linear(512, 512, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            sn(nn.Linear(512, 512, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            sn(nn.Linear(512, 512, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
        ])
        
        self.reshape = nn.Sequential(*[
            sn(nn.Linear(64, kt*base_hidden*4*4, bias=False)),
            layer.Reshape((-1, kt*base_hidden, 4, 4))
        ])
        
        self.const_input = const_input
        if const_input:
            self.input = ConstantInput(kt*base_hidden)

    def forward(self, z):
        z = self.projection(z)
        if self.const_input:
            x = self.input(z)
        else:
            x = self.reshape(z[:, :64])
            z = z[:, 64:]
        output, _ = self.main((x, z))
        return output

    
class block_d(nn.Module):
    def __init__(self, in_dim, out_dim, id=0):
        super().__init__()
        self.id = id
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        hidden_dim = in_dim // 4
        self.conv1 = sn(torch.nn.Conv2d(in_dim, hidden_dim, 1))
        self.conv2 = sn(torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
        self.conv3 = sn(torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1))
        self.conv4 = sn(torch.nn.Conv2d(hidden_dim, out_dim, 1))
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)
                
        self.conv_sc = sn(torch.nn.Conv2d(in_dim, out_dim - in_dim, 1))

    def forward(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(self.act(h))
        h = self.conv3(self.act(h))
        h = self.act(h)
        h = self.downsample(h)
        h = self.conv4(h)
        #shortcut
        x = self.downsample(x)
        x = torch.cat([x, self.conv_sc(x)], 1)
        return h+x

class ToFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        return torch.sum(self.act(x), [2,3])

# DCGAN discriminator (using somewhat the reverse of the generator)
class AutoGAN_D(torch.nn.Module):
    def __init__(self, img_size, base_hidden=16, pack=2, attention=64):
        super().__init__()
        self.pack = pack
        
        self.attention = False
        if attention is not None:
            self.attention = True
            self.attention_resolution = attention
        
        _block = block_d
        
        main = torch.nn.Sequential()
        ### Start block
        # Size = n_colors x image_size x image_size
        main.add_module('Start-block', sn(nn.Conv2d(3 * pack, base_hidden, 3, 1, 1)))
        image_size_new = img_size
        # Size = D_h_size x image_size/2 x image_size/2
        
        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 1
        while image_size_new > 1:
            if self.attention and image_size_new == self.attention_resolution:
                print('Attention!')
                main.add_module('Start-Attention', SelfAttention(base_hidden * mult))
            main.add_module('Millde-block [%d]' % i, _block(base_hidden * mult, base_hidden * (2*mult), id=i))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1
        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Feature', ToFeature())
        # Size = (bs, base_hidden * mult)
        self.out = nn.Sequential(*[
            #MinibatchDiscrimination(base_hidden*mult, base_hidden*mult+128),
            sn(nn.Linear(base_hidden*mult, 1))
        ])
        
        self.main = main

    def forward(self, input):
        bs, ch, w, h = input.shape
        input = input.reshape(bs//self.pack, ch*self.pack, w, h)
        feature = self.main(input)
        output = self.out(feature)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1, 1)
        
        
        
        
        
# =========================================================================

class DCGAN_G(torch.nn.Module):
    def __init__(self):
        super().__init__()
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = 64 // 8

        ### Start block
        # Z_size random numbers
        main.add_module('Start-ConvTranspose2d', sn(torch.nn.ConvTranspose2d(100, 128 * mult, kernel_size=4, stride=1, padding=0, bias=False)))
        main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(128 * mult))
        main.add_module('Start-ReLU', torch.nn.ReLU())
        # Size = (G_h_size * mult) x 4 x 4

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            main.add_module('Middle-ConvTranspose2d [%d]' % i, sn(torch.nn.ConvTranspose2d(128 * mult, 128 * (mult//2), kernel_size=4, stride=2, padding=1, bias=False)))
            main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(128 * (mult//2)))
            main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-ConvTranspose2d', sn(torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False)))
        main.add_module('End-TanH', torch.nn.Tanh())
        # Size = n_colors x image_size x image_size
        self.main = main
        self.reshape = layer.Reshape((-1, 100, 1, 1))

    def forward(self, input):
        input = self.reshape(input)
        output = self.main(input)
        return output

# DCGAN discriminator (using somewhat the reverse of the generator)
class DCGAN_D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        main = torch.nn.Sequential()

        ### Start block
        # Size = n_colors x image_size x image_size
        main.add_module('Start-Conv2d', sn(torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)))
        main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        image_size_new = 64 // 2
        # Size = D_h_size x image_size/2 x image_size/2

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            main.add_module('Middle-Conv2d [%d]' % i, sn(torch.nn.Conv2d(128 * mult, 128 * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(128 * (2*mult)))
            main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1

        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Conv2d', sn(torch.nn.Conv2d(128 * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, input):
        output = self.main(input)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1, 1)