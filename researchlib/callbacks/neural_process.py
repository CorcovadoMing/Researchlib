from .callback import Callback
import random
from ..utils import *
from torchvision.utils import save_image


class NeuralProcessCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_iteration_begin(self, **kwargs):
        if kwargs['model'].training:
            N = random.randint(1, 784)  # number of context points
        else:
            N = 300

        data = kwargs['data']
        batch_size = data.size(0)
        data = data.view(batch_size, -1, 1)
        context_idx = get_context_idx(N)
        x_context = idx_to_x(context_idx, batch_size)
        y_context = idx_to_y(context_idx, data)
        kwargs['data'] = [x_context, y_context, data]
        kwargs['target'] = data

        return kwargs

    def on_validation_end(self, **kwargs):
        data = kwargs['data'][2]
        batch_size = data.size(0)
        data = data.view(batch_size, -1, 1)

        plot_Ns = [10, 100, 300, 784]
        num_examples = min(batch_size, 16)

        for N in plot_Ns:
            recons = []
            context_idx = get_context_idx(N)
            x_context = idx_to_x(context_idx, batch_size)
            y_context = idx_to_y(context_idx, data)

            for d in range(5):
                y_hat, _, _ = kwargs['model'](x_context, y_context)
                recons.append(y_hat[:num_examples])

            recons = torch.cat(recons).view(-1, 1, 28,
                                            28).expand(-1, 3, -1, -1)
            background = torch.tensor([0., 0., 1.]).cuda()
            background = background.view(1, -1,
                                         1).expand(num_examples, 3,
                                                   784).contiguous()
            context_pixels = data[:num_examples].view(num_examples, 1,
                                                      -1)[:, :, context_idx]
            context_pixels = context_pixels.expand(num_examples, 3, -1)
            background[:, :, context_idx] = context_pixels
            comparison = torch.cat([background.view(-1, 3, 28, 28), recons])
            save_image(comparison.cpu(),
                       'ep_' + str(kwargs['epoch']) + '_cps_' + str(N) +
                       '.png',
                       nrow=num_examples)

        return kwargs
