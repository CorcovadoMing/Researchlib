import torch
import torch.nn.functional as F

def NormalDistributionKL(mu_q, logvar_q, mu_p, logvar_p):
    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
             - 1.0 \
             + logvar_p - logvar_q
    kl_div = 0.5 * kl_div.sum()
    return kl_div

def NeuralProcessLoss(output, target):
    y_hat, z_all, z_context = output
    y = target
    BCE = F.binary_cross_entropy(y_hat, y, reduction="sum")
    KLD = NormalDistributionKL(z_all[0], z_all[1], z_context[0], z_context[1])
    return BCE + KLD
