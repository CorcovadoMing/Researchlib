import torch.multiprocessing as mp
from IPython import display
import os


class ParallelExecutor:
    def __init__(self, task, max_job: int = 10, num_workers: int = os.cpu_count()):
        assert max_job > 0, 'max_job should be > 0'
        self.max_job = int(max_job)
        self.job_count = 0
        self.job = mp.Queue(self.max_job)
        self.reporter = mp.Queue(self.max_job)
        self.task = task
        self.num_workers = num_workers
        self.worker_pool = []
    
    def _worker(self, *args):
        while True:
            if not self.job.empty():
                desc = self.job.get()
                product = self.task(desc, *args)
                self.reporter.put(product)
        
    def put(self, job):
        self.job.put(job)
        self.job_count += 1
    
    def batch_put(self, batch_job):
        assert len(batch_job) <= self.max_job, 'Jobs are more than the queue, plase set larger max_job and try again'
        for job in batch_job:
            self.put(job)
    
    def get(self):
        return self.reporter.get()
    
    def empty(self):
        return self.reporter.empty()
    
    def start(self, *args):
        for rank in range(self.num_workers):
            p = mp.Process(target=self._worker, args=(*args,))
            p.start()
            self.worker_pool.append(p)
    
    def stop(self, *args):
        for i in self.worker_pool:
            i.terminate()
        self.worker_pool = []
    
    def wait(self):
        result = []
        while len(result) != self.job_count:
            result.append(self.get())
        self.job_count = 0
        display.clear_output(wait=True)
        return result
            