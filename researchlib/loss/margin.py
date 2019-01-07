import torch
from torch.autograd import Variable

def margin_loss(input, target, size_average=True):
    batch_size = input.size(0)

    # ||vc|| from the paper.
    v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))
    
    # Calculate left and right max() terms from equation 4 in the paper.
    zero = Variable(torch.zeros(1)).cuda()
    m_plus = 0.9
    m_minus = 0.1
    max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
    max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

    # This is equation 4 from the paper.
    loss_lambda = 0.5
    L_c = target * max_l + loss_lambda * (1.0 - target) * max_r
    L_c = L_c.sum(dim=1)

    if size_average:
        L_c = L_c.mean()
    else:
        L_c = L_c.sum()
    return L_c
