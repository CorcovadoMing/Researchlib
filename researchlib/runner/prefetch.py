import threading
import time
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


class BackgroundGenerator(threading.Thread):

    def __init__(self, generator, max_prefetch=2):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)

    def next(self):
        next_item = None
        while next_item is None:
            next_item = self.queue.get()
            self.queue.task_done()
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
    
    def __len__(self):
        return len(self.generator)
