from torch import nn
import torch
from ..utils import to_one_hot

class GANModel(nn.Module):
    def __init__(self, generator, discriminator, condition=False):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        if type(condition) == type([]) or type(condition) == type(()):
            self.g_condition, self.d_condition = condition
        else:
            self.g_condition, self.d_condition = condition, condition
    
    def set_optim(self, optim, lr=1e-3):
        self.optim_g = optim(self.generator.parameters(), lr=lr)
        self.optim_d = optim(self.discriminator.parameters(), lr=lr)
    
    def sample(self, bs, length=100, train=False, condition_data=None):
        noise = torch.randn(bs, length).cuda()
        if condition_data is not None:
            noise = torch.cat([noise, condition_data], dim=1)
        fake = self.generator(noise)
        if train:
            return fake
        else:
            return fake.detach()
    
    def forward_d(self, x, condition_data=None):
        if condition_data is not None:
            self.condition_data = to_one_hot(condition_data, 10).cuda()
        else:
            self.condition_data = None
            
        self.real_data = x
        if self.g_condition:
            self.fake_data = self.sample(x.size(0), condition_data=self.condition_data)
        else:
            self.fake_data = self.sample(x.size(0))
        
        if self.d_condition:
            fake = self.discriminator((self.fake_data, self.condition_data))
            real = self.discriminator((self.real_data, self.condition_data))
        else:
            fake = self.discriminator(self.fake_data)
            real = self.discriminator(self.real_data)
        return real, fake
    
    def forward_g(self, x, condition=None):
        if self.g_condition:
            fake = self.sample(x.size(0), train=True, condition_data=self.condition_data)
        else:
            fake = self.sample(x.size(0), train=True)
        
        if self.d_condition:
            return self.discriminator((fake, self.condition_data))
        else:
            return self.discriminator(fake)