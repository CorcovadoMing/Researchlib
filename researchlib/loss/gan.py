from torch import nn

def _wgan_d_loss(real, fake):
    return fake.mean() - real.mean()

def _wgan_g_loss(fake):
    return -fake.mean()

def _wgan_extra_step(model):
    for p in model.discriminator.parameters():
        p.data.clamp_(-0.05, 0.05)

class GANLoss(nn.Module):
    def __init__(self, arch='wgan'):
        super().__init__()
        if arch == 'wgan':
            self.d_loss = _wgan_d_loss
            self.g_loss = _wgan_g_loss
            self.extra_step = _wgan_extra_step
    
    def forward_d(self, real, fake):
        return self.d_loss(real, fake)
    
    def forward_g(self, fake):
        return self.g_loss(fake)
    
    def extra_step(self, model):
        self.extra_step(model)
        
        