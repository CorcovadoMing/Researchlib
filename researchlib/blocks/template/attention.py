import torch
from torch import nn
import torch.nn.functional as F
from ...layers import layer

class _SE_Attention(nn.Module):
    def __init__(self, dim, dim_type, ratio=16):
        super().__init__()
        self.fn = nn.Sequential(
            layer.__dict__['AdaptiveAvgPool' + dim_type](1),
            layer.__dict__['Conv' + dim_type](dim, dim // ratio, kernel_size=1), 
            nn.ReLU(inplace=True),
            layer.__dict__['Conv' + dim_type](dim // ratio, dim, kernel_size=1), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fn(x)

    
def _channel_pool(x):
    return torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1 )

    
class _CBAM_Attention(nn.Module):
    def __init__(self, dim, dim_type, ratio=16):
        super().__init__()
        
        # Channel
        self.max_pool = layer.__dict__['AdaptiveMaxPool' + dim_type](1)
        self.avg_pool = layer.__dict__['AdaptiveAvgPool' + dim_type](1)
        
        self.cfn = nn.Sequential(
            layer.__dict__['Conv' + dim_type](dim, dim // ratio, kernel_size=1), 
            nn.ReLU(inplace=True),
            layer.__dict__['Conv' + dim_type](dim // ratio, dim, kernel_size=1) 
        )
        
        # Spatial
        self.sfn = nn.Sequential(
            layer.__dict__['Conv' + dim_type](2, 1, 3, 1, 1),
            layer.__dict__['BatchNorm' + dim_type](1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        channel_attention = x * torch.sigmoid(self.cfn(self.max_pool(x)) + self.cfn(self.avg_pool(x)))
        spatial_attention = channel_attention * self.sfn(_channel_pool(channel_attention))
        return spatial_attention
        