from torch import nn
import torch
from ..utils import to_one_hot

class GANModel(nn.Module):
    def __init__(self, generator, discriminator, latent_vector_len=100, condition_vector_len=False, condition_onehot=False):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.condition_onehot = condition_onehot
        self.latent_vector_len = latent_vector_len
        if type(condition_vector_len) == list or type(condition_vector_len) == tuple:
            self.g_condition_vector_len, self.g_condition = self._parse_condition(condition_vector_len[0])
            self.d_condition_vector_len, self.d_condition = self._parse_condition(condition_vector_len[1])
        else:
            self.g_condition_vector_len, self.g_condition = self._parse_condition(condition_vector_len)
            self.d_condition_vector_len, self.d_condition = self._parse_condition(condition_vector_len)
    
    @staticmethod
    def _parse_condition(condition):
        _vector, _condition = 0, False
        if condition is not None and condition > 0: _vector, _condition = condition, True
        return _vector, _condition
    
    def set_optim(self, optim, lr=1e-3):
        self.optim_g = optim(self.generator.parameters(), lr=lr)
        self.optim_d = optim(self.discriminator.parameters(), lr=lr)
    
    def _parse_condition_data(self, condition_data, onehot, condition_vector):
        if type(condition_data) == range: condition_data = list(condition_data)
        if type(condition_data) == list or type(condition_data) == tuple: condition_data = torch.LongTensor(condition_data)
        device = condition_data.device
        condition_data = to_one_hot(condition_data.long(), condition_vector).cuda().float() if onehot else condition_data.float()
        if condition_data.dim() < 2: condition_data = condition_data.unsqueeze(-1)
        return condition_data.to(device)
    
    def sample(self, bs, condition_data=None, inference=True, requires_grad=False):
        noise = torch.empty((bs, self.latent_vector_len)).normal_(0, 1)
        if condition_data is not None:
            if inference: condition_data = self._parse_condition_data(condition_data, self.condition_onehot, self.g_condition_vector_len)
            noise = noise.to(condition_data.device)
            noise = torch.cat([noise, condition_data], dim=1)
        else:
            if not inference:
                noise = noise.cuda()
        fake = self.generator(noise)
        return fake if requires_grad else fake.detach()
    
    def forward_d(self, x, condition_data=None):
        if condition_data is not None:
            self.condition_data = self._parse_condition_data(condition_data, self.condition_onehot, self.g_condition_vector_len)
        else:
            self.condition_data = None
            
        self.real_data = x
        if self.g_condition:
            self.fake_data = self.sample(x.size(0), condition_data=self.condition_data, inference=False)
        else:
            self.fake_data = self.sample(x.size(0), inference=False)
        
        if self.d_condition:
            fake = self.discriminator((self.fake_data, self.condition_data))
            real = self.discriminator((self.real_data, self.condition_data))
        else:
            fake = self.discriminator(self.fake_data)
            real = self.discriminator(self.real_data)
        return real, fake
    
    def forward_g(self, x, condition=None):
        if self.g_condition:
            fake = self.sample(x.size(0), requires_grad=True, condition_data=self.condition_data, inference=False)
        else:
            fake = self.sample(x.size(0), requires_grad=True, inference=False)
        
        if self.d_condition:
            return self.discriminator((fake, self.condition_data))
        else:
            return self.discriminator(fake)