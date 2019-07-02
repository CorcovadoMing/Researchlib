import torch.nn.utils.spectral_norm as sn
import torch
from torch import nn
import torch.nn.functional as F
from ..layers import layer

class SelfModBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_f1 = sn(nn.Linear(64, num_features))
        self.beta_f1 = sn(nn.Linear(64, num_features))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_f1(y)
        beta = self.beta_f1(y)
        return gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

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
    
class block_g(nn.Module):
    def __init__(self, in_dim, out_dim, id=0, first=False):
        super().__init__()
        self.first = first
        self.id = id
        if not first:
            self.conv1 = sn(torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv2 = sn(torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False))
            self.conv_sc = sn(torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False))
        else:
            self.conv = sn(torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=1, padding=0, bias=False))
        self.bn1 = SelfModBatchNorm2d(in_dim)
        self.bn2 = SelfModBatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, input):
        x, y = input
        y_split = y[:, 64*(self.id+1):64*(self.id+2)]
        if not self.first:
            h = self.relu(self.bn1(x, y_split))
            h = self.upsample(h)
            x = self.upsample(x)
            x = self.conv_sc(x)
            h = self.conv1(h)
            h = self.relu(self.bn2(h, y_split))
            h = self.conv2(h)
            return (x+h, y)
        else:
            x = self.conv(x)
            return (x, y)

class ToRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Tanh()
    def forward(self, input):
        x, y = input
        return (self.f(x), y)

class AutoGAN_G(torch.nn.Module):
    def __init__(self, img_size, base_hidden=32):
        super().__init__()
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = img_size // 4

        ### Start block
        # Z_size random numbers
        main.add_module('Start-block', block_g(64, base_hidden*mult, id=0, first=True))
        # Size = (G_h_size * mult) x 4 x 4

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            main.add_module('Middle-block [%d]' % i, block_g(base_hidden*mult, base_hidden*(mult//2), id=i))
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-Att', SelfAttention(base_hidden))
        main.add_module('End-block', block_g(base_hidden, 3, id=i))
        main.add_module('End-Out', ToRGB())
        # Size = n_colors x image_size x image_size
        self.main = main
        self.reshape = layer.Reshape((-1, 64, 1, 1))

    def forward(self, z):
        input = self.reshape(z[:, :64])
        output, _ = self.main((input, z))
        return output


    
    
class block_d(nn.Module):
    def __init__(self, in_dim, out_dim, id=0):
        super().__init__()
        self.id = id
        self.conv1 = sn(torch.nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.conv2 = sn(torch.nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.downsample = nn.AvgPool2d(2)
        
        self.shortcut = nn.Sequential(*[
            sn(torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.AvgPool2d(2)
        ])
        
    def forward(self, x):
        h = self.conv1(self.lrelu(self.bn1(x)))
        h = self.conv2(self.lrelu(self.bn2(h)))
        h = self.downsample(h)
        return h + self.shortcut(x)

# DCGAN discriminator (using somewhat the reverse of the generator)
class AutoGAN_D(torch.nn.Module):
    def __init__(self, img_size, base_hidden=32):
        super().__init__()
        main = torch.nn.Sequential()

        ### Start block
        # Size = n_colors x image_size x image_size
        #main.add_module('Start-Conv2d', sn(torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)))
        #main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        main.add_module('Start-block', block_d(3, base_hidden, id=0))
        main.add_module('Start-Attention', SelfAttention(base_hidden))
        image_size_new = img_size // 2
        # Size = D_h_size x image_size/2 x image_size/2
        #main.add_module('Start-SelfAttention', SelfAttention(128))

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 1
        while image_size_new > 1:
            use_att = True if image_size_new > 2 else False
            main.add_module('Millde-block [%d]' % i, block_d(base_hidden * mult, base_hidden * (2*mult), id=i))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1
        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Conv2d', sn(torch.nn.Conv2d(base_hidden * mult, 1, kernel_size=1, stride=1, padding=0, bias=False)))
        # main.add_module('End-Sigmoid', torch.nn.Sigmoid())
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, input):
        output = self.main(input)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1, 1)
        