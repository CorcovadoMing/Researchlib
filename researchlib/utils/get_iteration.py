def _get_iteration(train_loader):
    iteration = None
    try:
        iteration = len(train_loader)
    except:
        iteration = (train_loader._size / train_loader.batch_size)
    assert iteration != None
    return iteration
