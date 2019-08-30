def inifinity_loop(loader):
    while True:
        for i, j in enumerate(loader):
            yield i, j
    