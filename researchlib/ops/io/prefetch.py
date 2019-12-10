import threading
import time
import sys
import torch

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


def _worker(generator, queue, fp16):
    for x, y in generator:
        if type(x) != torch.Tensor:
            x = torch.from_numpy(x)
        if type(y) != torch.Tensor:
            y = torch.from_numpy(y)

        x, y = x.pin_memory(), y.pin_memory()
        if fp16:
            x = x.half()
        x = x.cuda()
        y = y.cuda()

        queue.put((x, y))


class BackgroundGenerator:
    def __init__(self, generator, max_prefetch = 1, num_threads = 1, fp16 = False):
        self.queue = Queue.Queue(max_prefetch)
        self.fp16 = fp16
        self.generator = generator
        self.worker_thread = [
            threading.Thread(
                target = _worker, args = (self.generator, self.queue, self.fp16)
            ) for _ in range(num_threads)
        ]
        for i in self.worker_thread:
            i.start()

    def next(self):
        while self.queue.empty():
            # Simple spin lock
            time.sleep(0.001)
        x, y = self.queue.get()
        return x, y

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)
