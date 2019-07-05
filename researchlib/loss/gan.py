from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import torch

def _wgan_gp_d_loss(real, fake, *args):
    model = args[1]
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
    return F.binary_cross_entropy(torch.sigmoid(real), torch.ones(real.size(0), 1).cuda()) + F.binary_cross_entropy(torch.sigmoid(fake), torch.zeros(fake.size(0), 1).cuda())

def _vanilla_g_loss(fake, *args):
    return F.binary_cross_entropy(torch.sigmoid(fake), torch.ones(fake.size(0), 1).cuda())

def _lsgan_d_loss(real, fake, *args):
    return F.mse_loss(torch.sigmoid(real), torch.ones(real.size(0), 1).cuda()) + F.mse_loss(torch.sigmoid(fake), torch.zeros(fake.size(0), 1).cuda())
    
def _lsgan_g_loss(fake, *args):
    return F.mse_loss(torch.sigmoid(fake), torch.ones(fake.size(0), 1).cuda())

def _hinge_d_loss(real, fake, *args):
    return F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    
def _hinge_g_loss(fake, *args):
    return -fake.mean()

def _relative_vanilla_d_loss(real, fake, *args):
    rel = (real - fake)
    # Cache
    args[0].append(real)
    return F.binary_cross_entropy_with_logits(rel, torch.ones(rel.size(0), 1).cuda())

def _relative_vanilla_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    rel = (fake - real)
    return F.binary_cross_entropy_with_logits(rel, torch.ones(rel.size(0), 1).cuda())

def _relative_averaged_vanilla_d_loss(real, fake, *args):
    l1 = F.binary_cross_entropy_with_logits((real - fake.mean(0)), torch.ones(real.size(0), 1).cuda())
    l2 = F.binary_cross_entropy_with_logits((fake - real.mean(0)), torch.zeros(real.size(0), 1).cuda())
    # Cache
    args[0].append(real)
    return (l1+l2)/2

def _relative_averaged_vanilla_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    l1 = F.binary_cross_entropy_with_logits((real - fake.mean(0)), torch.zeros(real.size(0), 1).cuda())
    l2 = F.binary_cross_entropy_with_logits((fake - real.mean(0)), torch.ones(real.size(0), 1).cuda())
    return (l1+l2)/2

def _relative_averaged_lsgan_d_loss(real, fake, *args):
    l1 = torch.mean((real - fake.mean(0) - 1) ** 2)
    l2 = torch.mean((fake - real.mean(0) + 1) ** 2)
    # Cache
    args[0].append(real)
    return (l1+l2)/2

def _relative_averaged_lsgan_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    l1 = torch.mean((real - fake.mean(0) + 1) ** 2)
    l2 = torch.mean((fake - real.mean(0) - 1) ** 2)
    return (l1+l2)/2

def _relative_averaged_hinge_d_loss(real, fake, *args):
    l1 = F.relu(1.0 - (real - fake.mean(0))).mean()
    l2 = F.relu(1.0 + (fake - real.mean(0))).mean()
    # Cache
    args[0].append(real)
    return (l1+l2)/2

def _relative_averaged_hinge_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    l1 = F.relu(1.0 + (real - fake.mean(0))).mean()
    l2 = F.relu(1.0 - (fake - real.mean(0))).mean()
    return (l1+l2)/2

def _relative_centered_vanilla_d_loss(real, fake, *args):
    full_mean = (fake.mean(0) + real.mean(0)) / 2
    l1 = F.binary_cross_entropy_with_logits((real - full_mean), torch.ones(real.size(0), 1).cuda())
    l2 = F.binary_cross_entropy_with_logits((fake - full_mean), torch.zeros(real.size(0), 1).cuda())
    # Cache
    args[0].append(real)
    return (l1+l2)/2

def _relative_centered_vanilla_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    full_mean = (fake.mean(0) + real.mean(0)) / 2
    l1 = F.binary_cross_entropy_with_logits((real - full_mean), torch.zeros(real.size(0), 1).cuda())
    l2 = F.binary_cross_entropy_with_logits((fake - full_mean), torch.ones(real.size(0), 1).cuda())
    return (l1+l2)/2

def _relative_centered_lsgan_d_loss(real, fake, *args):
    full_mean = (fake.mean(0) + real.mean(0)) / 2
    l1 = torch.mean(((real - full_mean) - 1) ** 2)
    l2 = torch.mean(((fake - full_mean) + 1) ** 2)
    # Cache
    args[0].append(real)
    return (l1+l2)/2

def _relative_centered_lsgan_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    full_mean = (fake.mean(0) + real.mean(0)) / 2
    l1 = torch.mean(((real - full_mean) + 1) ** 2)
    l2 = torch.mean(((fake - full_mean) - 1) ** 2)
    return (l1+l2)/2

def _relative_centered_hinge_d_loss(real, fake, *args):
    full_mean = (fake.mean(0) + real.mean(0)) / 2
    l1 = F.relu(1 - (real - full_mean)).mean()
    l2 = F.relu(1 + (fake - full_mean)).mean()
    # Cache
    args[0].append(real)
    return (l1+l2)/2

def _relative_centered_hinge_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    full_mean = (fake.mean(0) + real.mean(0)) / 2
    l1 = F.relu(1 + (real - full_mean)).mean()
    l2 = F.relu(1 - (fake - full_mean)).mean()
    return (l1+l2)/2

def _relative_paired_vanilla_d_loss(real, fake, *args):
    grid_x, grid_y = torch.meshgrid([real.squeeze(), fake.squeeze()])
    subst = (grid_x - grid_y)
    # Cache
    args[0].append(real)
    return 2*F.binary_cross_entropy_with_logits(subst, torch.ones_like(subst).cuda())

def _relative_paired_vanilla_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    return F.binary_cross_entropy_with_logits(fake - real, torch.ones(real.size(0), 1).cuda())

def _relative_paired_lsgan_d_loss(real, fake, *args):
    grid_x, grid_y = torch.meshgrid([real.squeeze(), fake.squeeze()])
    subst = (grid_x - grid_y)
    # Cache
    args[0].append(real)
    return 2*torch.mean((subst - 1) ** 2)
    
def _relative_paired_lsgan_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    return torch.mean(((fake - real) - 1) ** 2)

def _relative_paired_hinge_d_loss(real, fake, *args):
    grid_x, grid_y = torch.meshgrid([real.squeeze(), fake.squeeze()])
    subst = (grid_x - grid_y)
    # Cache
    args[0].append(real)
    return 2*F.relu(1 - subst).mean()
    
def _relative_paired_hinge_g_loss(fake, *args):
    # Get cache from d_loss
    real = args[0].pop()
    return F.relu(1 - (fake - real)).mean()


def _wgan_extra_step(model, *args):
    for p in model.discriminator.parameters():
        p.data.clamp_(-0.1, 0.1)

def _noop_extra_step(model, *args):
    pass


class GANLoss(nn.Module):
    def __init__(self, arch='wgan', aux_loss=None, feature_match=True):
        super().__init__()
        self.feature_match = feature_match
        self.extra_step = _noop_extra_step
        self.aux_loss = aux_loss
        self.queue = []
        
        if arch == 'wgan':
            self.d_loss = _wgan_d_loss
            self.g_loss = _wgan_g_loss
            self.extra_step = _wgan_extra_step
        elif arch == 'wgan-gp':
            self.d_loss = _wgan_gp_d_loss
            self.g_loss = _wgan_gp_g_loss
        
        elif arch == 'vanilla':
            self.d_loss = _vanilla_d_loss
            self.g_loss = _vanilla_g_loss
        elif arch == 'lsgan':
            self.d_loss = _lsgan_d_loss
            self.g_loss = _lsgan_g_loss
        elif arch == 'hinge':
            self.d_loss = _hinge_d_loss
            self.g_loss = _hinge_g_loss
        
        elif arch == 'relative-vanilla':
            self.d_loss = _relative_vanilla_d_loss
            self.g_loss = _relative_vanilla_g_loss
        
        elif arch == 'relative-averaged-vanilla':
            self.d_loss = _relative_averaged_vanilla_d_loss
            self.g_loss = _relative_averaged_vanilla_g_loss
        elif arch == 'relative-averaged-lsgan':
            self.d_loss = _relative_averaged_lsgan_d_loss
            self.g_loss = _relative_averaged_lsgan_g_loss
        elif arch == 'relative-averaged-hinge':
            self.d_loss = _relative_averaged_hinge_d_loss
            self.g_loss = _relative_averaged_hinge_g_loss
            
        elif arch == 'relative-centered-vanilla':
            self.d_loss = _relative_centered_vanilla_d_loss
            self.g_loss = _relative_centered_vanilla_g_loss
        elif arch == 'relative-centered-lsgan':
            self.d_loss = _relative_centered_lsgan_d_loss
            self.g_loss = _relative_centered_lsgan_g_loss
        elif arch == 'relative-centered-hinge':
            self.d_loss = _relative_centered_hinge_d_loss
            self.g_loss = _relative_centered_hinge_g_loss
            
        elif arch == 'relative-paired-vanilla':
            self.d_loss = _relative_paired_vanilla_d_loss
            self.g_loss = _relative_paired_vanilla_g_loss
        elif arch == 'relative-paired-lsgan':
            self.d_loss = _relative_paired_lsgan_d_loss
            self.g_loss = _relative_paired_lsgan_g_loss
        elif arch == 'relative-paired-hinge':
            self.d_loss = _relative_paired_hinge_d_loss
            self.g_loss = _relative_paired_hinge_g_loss
        
        
    def set_model(self, model):
        self.model = model
    
    def forward_d(self, x, aux=None):
        real, fake = x
        if aux is not None:
            return self.aux_loss(real, aux) + self.aux_loss(fake, aux)
        else:
            return self.d_loss(real, fake, self.queue, self.model)
        
    def forward_g(self, fake, aux=None):
        if aux is not None:
            return self.aux_loss(*fake, aux)
        else:
            loss_extra = 0
            if self.feature_match:
                loss_extra += F.mse_loss(self.model.real_feature, self.model.fake_feature)
            return self.g_loss(fake, self.queue, self.model) + loss_extra
    
    def extra_step(self):
        self.extra_step(self.model)
        
        