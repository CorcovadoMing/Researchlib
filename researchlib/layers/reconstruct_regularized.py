from torch import nn
import torch
from torch.autograd import Variable

class ReconstructRegularized(nn.Module):
    def __init__(self, reconstruct_model, return_only_reconstruction=False):
        super().__init__()
        self.ror = return_only_reconstruction
        self.reconstruct_model = reconstruct_model
    
    def forward(self, x):
        if self.ror:
            return self.reconstruct_model(x)
        else:
            return x, self.reconstruct_model(x)
            
class CapReconstructRegularized(nn.Module):
    def __init__(self, reconstruct_model, return_only_reconstruction=False):
        super().__init__()
        self.ror = return_only_reconstruction
        self.reconstruct_model = reconstruct_model

    def forward(self, x):
        out = torch.norm(x, dim=-1)
        _, max_length_idx = out.max(dim=1)	
        y = torch.eye(10).cuda()
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)
        masked = (x*y).view(x.size(0), -1)

        if self.ror:
            return self.reconstruct_model(masked)
        else:
            return x, self.reconstruct_model(masked)