from torch import nn
import torch
import os


class _UpdateLabelsRecorder(nn.Module):
    def __init__(self, save_folder='recorder', averaging=10):
        super().__init__()
        self.ready_flag = False
        self.cache_list = {}
        self.averaging = averaging
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)
        self.idx = 0
        
    def forward(self, id_list, new_y):
        if self.training:
            with torch.no_grad():
                self.ready_flag = True 
                new_y = new_y.detach().cpu().clone()
                id_list = id_list.detach().long().cpu().clone().numpy()
                classes = new_y[0].size(-1)
                self.idx += 1
                self.idx %= self.averaging
                for i,j in zip(id_list, new_y):
                    if i not in self.cache_list:
                        self.cache_list[i] = torch.zeros(self.averaging, classes)
                    self.cache_list[i][self.idx, :] = j
                    
        else:
            if self.ready_flag:
                self.ready_flag = False
                ready_cache_list = {k: v.mean(0) for k, v in self.cache_list.items()}
                torch.save(ready_cache_list, os.path.join(self.save_folder, 'soft_label.pt'))
                ready_cache_list = {k: v.mean(0).argmax(-1) for k, v in self.cache_list.items()}
                torch.save(ready_cache_list, os.path.join(self.save_folder, 'hard_label.pt'))