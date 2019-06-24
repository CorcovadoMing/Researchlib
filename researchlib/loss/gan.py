from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import torch

def _wgan_gp_d_loss(real, fake, *args):
    model = args[0]
    alpha = torch.rand((real.size(0), 1, 1, 1)).cuda()
    x_hat = alpha * model.real_data + (1 - alpha) * model.fake_data
    x_hat.requires_grad = True

    if model.d_condition:
        pred_hat = model.discriminator((x_hat, model.condition_data))
    else:    
        pred_hat = model.discriminator(x_hat)
    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = 10 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2)
    return fake.mean() - real.mean() + gradient_penalty.mean()

def _wgan_gp_g_loss(fake, *args):
    return -fake.mean()


def _wgan_d_loss(real, fake, *args):
    return fake.mean() - real.mean()

def _wgan_g_loss(fake, *args):
    return -fake.mean()

def _vanilla_d_loss(real, fake, *args):
    return F.binary_cross_entropy(real, torch.ones(real.size(0), 1).cuda()-0.1) + F.binary_cross_entropy(fake, torch.zeros(fake.size(0), 1).cuda()+0.1)

def _vanilla_g_loss(fake, *args):
    return F.binary_cross_entropy(fake, torch.ones(fake.size(0), 1).cuda()-0.1)

def _lsgan_d_loss(real, fake, *args):
    return F.mse_loss(real, torch.ones(real.size(0), 1).cuda()) + F.mse_loss(fake, torch.zeros(fake.size(0), 1).cuda())
    
def _lsgan_g_loss(fake, *args):
    return F.mse_loss(fake, torch.ones(fake.size(0), 1).cuda())


def _wgan_extra_step(model, *args):
    for p in model.discriminator.parameters():
        p.data.clamp_(-0.1, 0.1)

def _noop_extra_step(model, *args):
    pass


class GANLoss(nn.Module):
    def __init__(self, arch='wgan', aux_loss=None):
        super().__init__()
        self.extra_step = _noop_extra_step
        self.aux_loss = aux_loss
        if arch == 'wgan':
            self.d_loss = _wgan_d_loss
            self.g_loss = _wgan_g_loss
            self.extra_step = _wgan_extra_step
        if arch == 'wgan-gp':
            self.d_loss = _wgan_gp_d_loss
            self.g_loss = _wgan_gp_g_loss
        if arch == 'vanilla':
            self.d_loss = _vanilla_d_loss
            self.g_loss = _vanilla_g_loss
        if arch == 'lsgan':
            self.d_loss = _lsgan_d_loss
            self.g_loss = _lsgan_g_loss
        
    def set_model(self, model):
        self.model = model
    
    def forward_d(self, x, aux=None):
        real, fake = x
        if aux is not None:
            return self.aux_loss(real, aux) + self.aux_loss(fake, aux)
        else:
            return self.d_loss(real, fake, self.model)
        
    def forward_g(self, fake, aux=None):
        if aux is not None:
            return self.aux_loss(*fake, aux)
        else:
            return self.g_loss(fake, self.model)
    
    def extra_step(self):
        self.extra_step(self.model)
        
        