def normalize(x, mean, std):
    x -= mean
    x /= std
    return x