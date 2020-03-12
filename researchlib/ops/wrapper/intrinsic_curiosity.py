from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.init as init


class _IntrinsicCuriosity(nn.Module):
    def __init__(self, encoder, inverse_model, forward_model, beta = 0.2):
        super().__init__()
        self.beta = beta
        self.encoder = encoder
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        self.device = None
        
    def forward(self, x):
        if self.device is None:
            self.device = next(self.encoder.parameters()).device
        
        if type(x) == list: # RL
            for i in range(len(x)):
                for j in range(len(x[i])):
                    state = torch.stack(x[i][j]['state'],  0).to(self.device)
                    state_encoding = self.encoder(state)
                    next_state_encoding = torch.cat([state_encoding[1:], 
                                                     torch.zeros((1, state_encoding.size(1))).to(self.device)])
                    
                    # forward model branch
                    action = torch.LongTensor(x[i][j]['action']).to(self.device)
                    pred_next_state_encoding = self.forward_model(state_encoding, action)
                    x[i][j]['intrinsic'] = (pred_next_state_encoding - next_state_encoding).detach().abs().sum(-1).cpu()
                    
                    # inverse model branch
                    pred_action = self.inverse_model(state_encoding, next_state_encoding)
                    
                    x[i][j]['intrinsic_loss'] = (1-self.beta) * F.nll_loss(pred_action, 
                                                                           action.detach(), 
                                                                           reduction='none') \
                                                + self.beta * \
                                                (pred_next_state_encoding - next_state_encoding.detach()).abs().sum(-1)
            return x
        
        