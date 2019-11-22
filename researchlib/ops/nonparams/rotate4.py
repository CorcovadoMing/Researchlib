from torch import nn
import torch


r0 = lambda x: x
r90 = lambda x: x.transpose(-2, -1)
r180 = lambda x: x.flip(-2)
r270 = lambda x: x.transpose(-2, -1).flip(-1)
func_list = [r0, r90, r180, r270]

class _Rotation42d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _get_process_fn(self, angle):
        return func_list[angle]
        
    def forward(self, x):
        if self.training:
            with torch.no_grad():
                x_chunk = torch.chunk(x, 4)
                new_x = []
                new_y = []
                for i, data in enumerate(x_chunk):
                    new_x.append(func_list[i](data))
                    new_y.append(torch.zeros([data.size(0)], device=x.device).fill_(i).long())
            return torch.cat(new_x, dim=0), torch.cat(new_y, dim=0)
        else:
            return x, torch.zeros([x.size(0)], device=x.device).long()