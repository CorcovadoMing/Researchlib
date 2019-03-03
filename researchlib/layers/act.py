import torch
from torch import nn

class ACTCell(nn.Module):
    def __init__(self, base_cell, in_dim, hidden_dim, out_dim, max_ponder=100, epison=0.01):
        super().__init__()
        self.in_dim = in_dim + 1
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.base_cell = base_cell(self.in_dim, self.hidden_dim)
        self.ponder_linear = nn.Linear(self.hidden_dim, 1)
        self.out_linear = nn.Linear(self.hidden_dim, self.out_dim)
        self.max_ponder = max_ponder
        self.epison = epison
        
    def forward(self, x, state):
        # Initialize
        hx, cx = state
        batch_size = x.size()[0]
        ponder_count = 0
        
        gx_acc = torch.zeros(batch_size, 1).cuda()
        aggregate_hx = torch.zeros(batch_size, self.hidden_dim).cuda()
        aggregate_cx = torch.zeros(batch_size, self.hidden_dim).cuda()
        aggregate_out = torch.zeros(batch_size, self.out_dim).cuda()
        binary_flag = torch.ones(batch_size, 1).cuda()

        while ponder_count < self.max_ponder:
            if ponder_count == 1:
                binary_flag = 1 - binary_flag

            hx, cx = self.base_cell(torch.cat((x, binary_flag), dim=1), (hx, cx))

            # Gates
            gx = torch.sigmoid(self.ponder_linear(hx))
            halt_prev = gx_acc > (1 - self.epison)
            gx_acc += (gx * (1 - halt_prev.float()))
            halt_after = gx_acc > (1 - self.epison)
            halt = halt_after - halt_prev
            rx = (gx_acc - 1) * halt.float()
            p = (gx * (1 - halt_prev.float())) - rx
            gx_acc -= rx
            ponder_count += 1
            aggregate_hx += p * hx
            aggregate_cx += p * cx
            aggregate_out += p * self.out_linear(hx)

            if halt_after.sum() == batch_size:
                break
                
        return (aggregate_hx, aggregate_cx), aggregate_out, ponder_count

    
    
class ACT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.cell = ACTCell(nn.LSTMCell, in_dim, hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        hx = torch.zeros(x.size()[0], self.hidden_dim).cuda()
        cx = torch.zeros(x.size()[0], self.hidden_dim).cuda()
        ponder_count = 0.
        for i in range(x.size()[2]):
            (hx, cx), out, pc = self.cell(x[:, :, i], (hx, cx))
            ponder_count += pc
        return out