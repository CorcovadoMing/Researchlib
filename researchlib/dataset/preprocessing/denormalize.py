def denormalize(x, mean, std):
    x *= std
    x += mean
    return x