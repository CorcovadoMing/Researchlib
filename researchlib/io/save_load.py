import torch

def save_model(model, path):
    if type(model) == type([]):
        prefix = path.split('.h5')[0]
        for i, m in enumerate(model):
            new_path = prefix + str(i) + '.h5'
            torch.save(m.state_dict(), new_path)
    else:
        torch.save(model.state_dict(), path)

def load_model(model, path):
    if type(model) == type([]):
        prefix = path.split('.h5')[0]
        for i, m in enumerate(model):
            new_path = predix + str(i) + '.h5'
            sd = torch.load(new_path, map_location=lambda storage, loc: storage)
            names = set(m.state_dict().keys())
            for n in list(sd.keys()): # list "detatches" the iterator
                if n not in names and n+'_raw' in names:
                    if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
                    del sd[n]
            m.load_state_dict(sd)
    else:
        sd = torch.load(path, map_location=lambda storage, loc: storage)
        names = set(model.state_dict().keys())
        for n in list(sd.keys()): # list "detatches" the iterator
            if n not in names and n+'_raw' in names:
                if n+'_raw' not in sd: sd[n+'_raw'] = sd[n]
                del sd[n]
        model.load_state_dict(sd)