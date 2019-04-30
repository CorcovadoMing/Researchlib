from torch import nn
import torch 

class GANModel(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    def set_optim(self, optim, lr=1e-3):
        self.optim_g = optim(self.generator.parameters(), lr=lr)
        self.optim_d = optim(self.discriminator.parameters(), lr=lr)
    
    def sample(self, bs, length=100, train=False):
        noise = torch.randn(bs, length).cuda()
        if train:
            return self.generator(noise)
        else:
            return self.generator(noise).detach()
    
    def forward_d(self, x):
        fake = self.discriminator(self.sample(x.size(0)))
        real = self.discriminator(x)
        return real, fake
    
    def forward_g(self, x):
        return self.discriminator(self.sample(x.size(0), train=True))