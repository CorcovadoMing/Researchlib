import math


class _NumpyIterator(object):

    def __init__(self, batch_size, x, y):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        self.i = 0
        self.n = math.floor(len(self.x) / self.batch_size)
        return self

    def __next__(self):
        x = self.x[int(self.i * self.batch_size):int((self.i + 1) *
                                                     self.batch_size)]
        y = self.y[int(self.i * self.batch_size):int((self.i + 1) *
                                                     self.batch_size)]
        self.i = (self.i + 1) % self.n

        return x[:, :, :, None], y[:, :, :, None]

    next = __next__


from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator


class _AugPipeline(Pipeline):

    def __init__(self, iterator, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.int32 = ops.Cast(device="gpu", dtype=types.INT32)
        self.float = ops.Cast(device="gpu", dtype=types.FLOAT)
        self.flip = ops.Flip(device="gpu", vertical=0, horizontal=1)
        self.iterator = iterator

    def iter_setup(self):
        images, labels = self.iterator.next()
        self.feed_input(self.x, images)
        self.feed_input(self.y, labels)

    def define_graph(self):
        self.x = self.input().gpu()
        self.y = self.input_label().gpu()
        x = self.float(self.x)
        y = self.float(self.y)
        output_img = self.flip(x)
        output_label = self.flip(y)
        return (output_img, output_label)


def FromDali(x, y, batch_size=1, num_workers=4):
    pipe = _AugPipeline(
        iter(_NumpyIterator(batch_size, x, y)),
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0)
    pipe.build()
    return DALIGenericIterator(pipe, ['data', 'label'], len(x), auto_reset=True)
