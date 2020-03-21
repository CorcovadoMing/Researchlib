import threading
import time
import sys
import torch

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


def _worker(generator, queue, fp16):
    for data in generator:
        data = [torch.from_numpy(i) if type(i) != torch.Tensor else i for i in data]
        if fp16:
            data = [j.half() if i != 1 else j for i, j in enumerate(data)]
        data = [i.cuda(non_blocking=True) for i in data]
        queue.put(data)


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
        return self.queue.get()

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)
