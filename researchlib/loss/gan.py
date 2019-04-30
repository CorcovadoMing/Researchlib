from torch import nn
from torch.autograd import grad
import torch

def _wgan_gp_d_loss(real, fake, *args):
    alpha = torch.rand((real.size(0), 1, 1, 1)).cuda()
    x_hat = alpha * args[0].real_data + (1 - alpha) * args[0].fake_data
    x_hat.requires_grad = True

    pred_hat = args[0].discriminator(x_hat)
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

def _wgan_extra_step(model, *args):
    for p in model.discriminator.parameters():
        p.data.clamp_(-0.05, 0.05)

def _noop_extra_step(model, *args):
    pass


class GANLoss(nn.Module):
    def __init__(self, arch='wgan'):
        super().__init__()
        if arch == 'wgan':
            self.d_loss = _wgan_d_loss
            self.g_loss = _wgan_g_loss
            self.extra_step = _wgan_extra_step
        if arch == 'wgan-gp':
            self.d_loss = _wgan_gp_d_loss
            self.g_loss = _wgan_gp_g_loss
            self.extra_step = _noop_extra_step
    
    def set_model(self, model):
        self.model = model
    
    def forward_d(self, x):
        real, fake = x
        return self.d_loss(real, fake, self.model)
    
    def forward_g(self, fake):
        return self.g_loss(fake, self.model)
    
    def extra_step(self):
        self.extra_step(self.model)
        
        