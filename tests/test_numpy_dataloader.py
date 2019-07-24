from researchlib.single_import import *

def test_numpy_loader():
    x = np.random.rand(10, 3, 32, 32)
    y = np.random.rand(10)
    for i in range(1, 10):
        data_loader = FromNumpy((x, y), 1, batch_size=i)
        x_, y_ = next(iter(data_loader[0]))
        assert x_.shape == (i, 3, 32, 32)
        assert y_.shape == (i,)
