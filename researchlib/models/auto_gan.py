import torch.nn.utils.spectral_norm as sn
import torch
from torch import nn
import torch.nn.functional as F
from ..layers import layer

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
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma_f1 = sn(nn.Linear(64, 32))
        self.gamma_f2 = sn(nn.Linear(32, num_features))
        self.beta_f1 = sn(nn.Linear(64, 32))
        self.beta_f2 = sn(nn.Linear(32, num_features))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_f2(F.relu(self.gamma_f1(y)))
        beta = self.beta_f2(F.relu(self.beta_f1(y)))
        return (1+gamma).view(out.size(0), -1, 1, 1) * out + beta.view(out.size(0), -1, 1, 1)

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
    def __init__(self, in_dim, out_dim, id=0):
        super().__init__()
        self.id = id
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        hidden_dim = in_dim // 4
        self.conv1 = sn(torch.nn.Conv2d(in_dim, hidden_dim, 1, bias=False))
        self.conv2 = sn(torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
        self.conv3 = sn(torch.nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=False))
        self.conv4 = sn(torch.nn.Conv2d(hidden_dim, out_dim, 1, bias=False))
        
        self.bn1 = SelfModNorm2d(in_dim)
        self.bn2 = SelfModNorm2d(hidden_dim)
        self.bn3 = SelfModNorm2d(hidden_dim)
        self.bn4 = SelfModNorm2d(hidden_dim)        
        
        self.act = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, input):
        x, y = input
        
        h = self.conv1(self.act(self.bn1(x, y[:, self.id*64:(self.id+1)*64])))
        h = self.act(self.bn2(h, y[:, self.id*64:(self.id+1)*64]))
        
        if self.in_dim != self.out_dim:
            # Drop channels
            x = x[:, :self.out_dim]
        
        h = self.upsample(h)
        x = self.upsample(x)
        
        h = self.conv2(h)
        h = self.conv3(self.act(self.bn3(h, y[:, self.id*64:(self.id+1)*64])))
        h = self.conv4(self.act(self.bn4(h, y[:, self.id*64:(self.id+1)*64])))
        return (h+x, y)

class ToRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Tanh()
    def forward(self, input):
        x, y = input
        return (self.f(x), y)

class AutoGAN_G(torch.nn.Module):
    def __init__(self, img_size, base_hidden=16):
        super().__init__()
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = img_size // 8
        
        i = 0
        while mult > 1:
            main.add_module('Middle-block [%d]' % i, block_g(base_hidden*mult, base_hidden*(mult//2), id=i))
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        print(i)
        
        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-Att', SelfAttention(base_hidden))
        main.add_module('End-block', block_g(base_hidden, 3, id=i))
        main.add_module('End-Out', ToRGB())
        # Size = n_colors x image_size x image_size
        self.main = main
        self.reshape = nn.Sequential(*[
            sn(nn.Linear(64, 16*base_hidden*4*4)),
            layer.Reshape((-1, 16*base_hidden, 4, 4))
        ])

    def forward(self, z):
        input = self.reshape(z[:, :64])
        output, _ = self.main((input, z[:, 64:]))
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
        
        self.act = nn.LeakyReLU(0.2)
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
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        return torch.sum(self.act(x), [2,3])

# DCGAN discriminator (using somewhat the reverse of the generator)
class AutoGAN_D(torch.nn.Module):
    def __init__(self, img_size, base_hidden=16, pack=2):
        super().__init__()
        self.pack = pack
        
        main = torch.nn.Sequential()
        ### Start block
        # Size = n_colors x image_size x image_size
        #main.add_module('Start-Conv2d', sn(torch.nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False)))
        #main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        main.add_module('Start-block', sn(nn.Conv2d(3 * pack, base_hidden, 3, 1, 1)))
        image_size_new = img_size
        # Size = D_h_size x image_size/2 x image_size/2
        #main.add_module('Start-SelfAttention', SelfAttention(128))

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 1
        while image_size_new > 1:
            if image_size_new == 64:
                print('Attention!')
                main.add_module('Start-Attention', SelfAttention(base_hidden * mult))
            main.add_module('Millde-block [%d]' % i, block_d(base_hidden * mult, base_hidden * (2*mult), id=i))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1
        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Feature', ToFeature())
        # Size = (bs, base_hidden * mult)
        self.out = nn.Sequential(*[
#             MinibatchDiscrimination(base_hidden*mult, base_hidden*mult+32),
            sn(nn.Linear(base_hidden*mult, 1))
        ])
        
        self.main = main

    def forward(self, input):
        bs, ch, w, h = input.shape
        input = input.reshape(bs//self.pack, ch*self.pack, w, h)
        feature = self.main(input)
        output = self.out(feature)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1, 1), feature
        