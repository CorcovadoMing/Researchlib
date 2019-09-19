import threading
import time
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


def _worker(generator, queue):
    for item in generator:
        queue.put(item)


class BackgroundGenerator:
    def __init__(self, generator, max_prefetch = 2):
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.worker_thread = threading.Thread(
            target = _worker, args = (self.generator, self.queue)
        )
        self.worker_thread.start()

    def next(self):
        while self.queue.empty():
            time.sleep(0.01)  # SPIN LOCK
        return self.queue.get()

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)
