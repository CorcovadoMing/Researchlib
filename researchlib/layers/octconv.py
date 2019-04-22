import torch.nn as nn
import torch.nn.functional as F

class OctConv2d(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, padding=0, alphas=(0.5, 0.5)):
        super().__init__()
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_in <= 1, "Alphas must be in interval [0, 1]"

        # CH IN
        self.ch_in_hf = int((1 - self.alpha_in) * ch_in)
        self.ch_in_lf = ch_in - self.ch_in_hf

        # CH OUT
        self.ch_out_hf = int((1 - self.alpha_out) * ch_out)
        self.ch_out_lf = ch_out - self.ch_out_hf
        
        try:
            self.H2H = nn.Conv2d(self.ch_in_hf, self.ch_out_hf, kernel_size, stride, padding)
        except:
            pass
        
        try:
            self.H2L = nn.Conv2d(self.ch_in_hf, self.ch_out_lf, kernel_size, stride, padding)
        except:
            pass
        
        try:
            self.L2H = nn.Conv2d(self.ch_in_lf, self.ch_out_hf, kernel_size, stride, padding)
        except: 
            pass
        
        try:
            self.L2L = nn.Conv2d(self.ch_in_lf, self.ch_out_lf, kernel_size, stride, padding)
        except:
            pass
        
    def forward(self, x):
        if self.alpha_in == 0 and self.alpha_out == 0:
            # Normal convolution
            return self.H2H(x)
        
        elif self.alpha_in == 0 and self.alpha_out != 0:
            # First layer, only high freq input, but with H and L outputs
            out_h = self.H2H(x)
            out_l = self.H2L(F.avg_pool2d(x, 2))
            return out_h, out_l
            
        elif self.alpha_in != 0 and self.alpha_out != 0:
            # Both H and L inputs and outputs
            in_h, in_l = x
            h2h = self.H2H(in_h)
            h2l = F.avg_pool2d(self.H2L(in_h), 2)
            l2h = F.interpolate(self.L2H(in_l), h2h.shape[-1])
            l2l = self.L2L(in_l)
            out_h = h2h + l2h
            out_l = h2l + l2l
            return out_h, out_l
        
        else:
            # Both H and L inputs, only H outputs
            in_h, in_l = x
            h2h = self.H2H(in_h)
            l2h = F.interpolate(self.L2H(in_l), h2h.shape[-1])
            out_h = h2h + l2h
            return out_h