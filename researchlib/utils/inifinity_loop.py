def inifinity_loop(loader, epoch=-1):
    count = 0
    while True:
        for i, j in enumerate(loader):
            yield i, j
        count += 1
        if count == epoch:
            break
