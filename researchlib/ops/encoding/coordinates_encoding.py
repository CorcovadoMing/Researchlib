from torch import nn
import torch


class _CoordinatesEncoding2d(nn.Module):
    def __init__(self, with_r = False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x):
        batch_size, _, image_height, image_width = x.size()
        y_coords = torch.arange(image_height).unsqueeze(1).expand(image_height, image_width).float() / image_height
        x_coords = torch.arange(image_width).unsqueeze(0).expand(image_height, image_width).float() / image_width
        coords = torch.stack((y_coords, x_coords), dim = 0)
        if self.with_r:
            rs = ((y_coords ** 2) + (x_coords ** 2)) ** 0.5
            rs = torch.unsqueeze(rs, dim = 0)
            coords = torch.cat((coords, rs), dim = 0)

        coords = torch.unsqueeze(coords, dim = 0).repeat(batch_size, 1, 1, 1).float()
        return torch.cat((x, coords.to(x.device)), dim = 1)