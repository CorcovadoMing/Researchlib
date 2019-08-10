def mapping(value, source, target, to_int=False):
    smin, smax = source[0], source[1]
    srange = smax - smin
    tmin, tmax = target[0], target[1]
    trange = tmax - tmin

    vratio = (value - smin) / srange
    target_value = (trange * vratio) + tmin

    if to_int:
        target_value = int(target_value)

    return target_value
