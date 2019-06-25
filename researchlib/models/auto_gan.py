import torch.nn.utils.spectral_norm as sn
import torch
from ..layers import layer

class AutoGAN_G(torch.nn.Module):
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
class AutoGAN_D(torch.nn.Module):
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
        # main.add_module('End-Sigmoid', torch.nn.Sigmoid())
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, input):
        output = self.main(input)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1, 1)