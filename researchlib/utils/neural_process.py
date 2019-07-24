import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_context_idx(N):
    # generate the indeces of the N context points in a flattened image
    idx = random.sample(range(0, 784), N)
    idx = torch.tensor(idx, device=device)
    return idx


def generate_grid(h, w):
    rows = torch.linspace(0, 1, h, device=device)
    cols = torch.linspace(0, 1, w, device=device)
    grid = torch.stack(
        [cols.repeat(h, 1).t().contiguous().view(-1),
         rows.repeat(w)], dim=1)
    grid = grid.unsqueeze(0)
    return grid


def idx_to_y(idx, data):
    # get the [0;1] pixel intensity at each index
    y = torch.index_select(data, dim=1, index=idx)
    return y


def idx_to_x(idx, batch_size):
    # From flat idx to 2d coordinates of the 28x28 grid. E.g. 35 -> (1, 7)
    # Equivalent to np.unravel_index()
    x = torch.index_select(generate_grid(28, 28), dim=1, index=idx)
    x = x.expand(batch_size, -1, -1)
    return x
