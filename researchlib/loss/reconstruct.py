import torch

def reconstruction_loss(input, target, size_average=True):
    # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
    # Multiplied by a small number so it doesn't dominate the margin (class) loss.
    error = (input - target).view(input.size(0), -1)
    error = error**2
    error = torch.sum(error, dim=1)

    # Average over batch
    if size_average:
        error = error.mean()
    else:
        error = error.sum()

    return error
