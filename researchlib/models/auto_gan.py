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
        self.gamma_f1 = sn(nn.Linear(100, 32))
        self.gamma_f2 = sn(nn.Linear(32, num_features, bias=False))
        self.beta_f1 = sn(nn.Linear(100, 32))
        self.beta_f2 = sn(nn.Linear(32, num_features, bias=False))

    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma_f2(F.relu(self.gamma_f1(y)))
        beta = self.beta_f2(F.relu(self.beta_f1(y)))
        return gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = sn(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.key_conv = sn(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.value_conv = sn(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C X (N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N
        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        out = self.gamma*out + x
        return out

class block_g(nn.Module):
    def __init__(self, in_dim, out_dim, scale=2):
        super().__init__()
        self.scale = scale
        if scale == 2:
            self.conv = sn(torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False))
            self.shortcut = nn.Sequential(*[
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                sn(torch.nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False))
            ])
        else:
            self.conv = sn(torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=1, padding=0, bias=False))
        self.bn = SelfModBatchNorm2d(out_dim)
        #self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    def forward(self, input):
        x, y = input
        if self.scale == 2:
            s = self.shortcut(x)
            x = self.conv(x)
            x = self.bn(x, y)
            #x = self.bn(x)
            x = self.relu(x)
            return (x+s, y)
        else:
            x = self.conv(x)
            x = self.bn(x, y)
            #x = self.bn(x)
            x = self.relu(x)
            return (x, y)

class final_block_g(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.attn = SelfAttention(in_dim)
        self.conv = sn(torch.nn.ConvTranspose2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False))
        self.out = torch.nn.Tanh()
    def forward(self, input):
        x, y = input
        x = self.attn(x)
        x = self.conv(x)
        x = self.out(x)
        return (x, y)
        
class AutoGAN_G(torch.nn.Module):
    def __init__(self, img_size, base_hidden=128):
        super().__init__()
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = img_size // 8

        ### Start block
        # Z_size random numbers
        main.add_module('Start-block', block_g(100, base_hidden*mult, scale=4))
        # Size = (G_h_size * mult) x 4 x 4

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            main.add_module('Middle-block [%d]' % i, block_g(base_hidden*mult, base_hidden*(mult//2)))
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        
        main.add_module('End-block', final_block_g(base_hidden, 3))
        # Size = n_colors x image_size x image_size
        self.main = main
        self.reshape = layer.Reshape((-1, 100, 1, 1))

    def forward(self, z):
        input = self.reshape(z)
        output, _ = self.main((input, z))
        return output


    
    
class block_d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = sn(torch.nn.Conv2d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn = nn.BatchNorm2d(out_dim)
        self.lrelu = nn.LeakyReLU(0.2)
#         self.shortcut = nn.Sequential(*[
#             nn.AvgPool2d(2),
#             sn(nn.Conv2d(in_dim, out_dim, 1))
#         ])
    def forward(self, x):
        #s = self.shortcut(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x#+s

# DCGAN discriminator (using somewhat the reverse of the generator)
class AutoGAN_D(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()
        main = torch.nn.Sequential()

        ### Start block
        # Size = n_colors x image_size x image_size
        main.add_module('Start-Conv2d', sn(torch.nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1, bias=False)))
        main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        image_size_new = img_size // 2
        # Size = D_h_size x image_size/2 x image_size/2
        # main.add_module('Start-SelfAttention', SelfAttention(128))

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            main.add_module('Millde-block [%d]' % i, block_d(128 * mult, 128 * (2*mult)))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1

        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Conv2d', sn(torch.nn.Conv2d(128 * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
        # main.add_module('End-Sigmoid', torch.nn.Sigmoid())
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, input):
        output = self.main(input)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1, 1)
        