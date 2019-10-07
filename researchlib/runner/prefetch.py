import threading
import time
import sys
import torch

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


def _worker(generator, queue, stream):
    for x, y in generator:
        
        if type(x) != torch.Tensor:
            x = torch.from_numpy(x)
        if type(y) != torch.Tensor:
            y = torch.from_numpy(y)
        
        x, y = x.pin_memory(), y.pin_memory()
        
        with torch.cuda.stream(stream):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            
        queue.put((x, y))
        
        
class BackgroundGenerator:
    def __init__(self, generator, max_prefetch = 2, num_threads = 2):
        self.queue = Queue.Queue(max_prefetch)
        self.stream = torch.cuda.Stream()
        self.generator = generator
        self.worker_thread = [threading.Thread(target = _worker, args = (self.generator, self.queue, self.stream)) for _ in range(num_threads)]
        for i in self.worker_thread:
            i.start()

    def next(self):
        while self.queue.empty():
            # Simple spin lock
            time.sleep(0.001)
        x, y = self.queue.get()
        torch.cuda.current_stream().wait_stream(self.stream)
        return x, y

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)
