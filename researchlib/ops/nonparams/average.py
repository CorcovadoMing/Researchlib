def _Average(*x):
    accum = 0
    count = 0
    for i in x:
        if i is not None:
            accum += i
            count += 1
    return accum / count