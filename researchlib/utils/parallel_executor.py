import torch.multiprocessing as mp
from IPython import display

class ParallelExecutor:
    def __init__(self, task, max_job: int = 10, num_workers: int = 8):
        assert max_job > 0
        self.max_job = int(max_job)
        self.job_count = 0
        self.job = mp.Queue(self.max_job)
        self.reporter = mp.Queue(self.max_job)
        self.task = task
        self.num_workers = num_workers
    
    def _worker(self, *args):
        while not self.job.empty():
            desc = self.job.get()
            product = self.task(desc, args)
            self.reporter.put(product)
        
    def put(self, job):
        self.job.put(job)
        self.job_count += 1
    
    def batch_put(self, batch_job):
        for job in batch_job:
            self.put(job)
    
    def get(self):
        return self.reporter.get()
    
    def empty(self):
        return self.reporter.empty()
    
    def start(self, *args):
        for rank in range(self.num_workers):
            p = mp.Process(target=self._worker, args=(args,))
            p.start()
    
    def wait(self):
        result = []
        while len(result) != self.job_count:
            result.append(self.get())
        self.job_count = 0
        display.clear_output(wait=True)
        return result
            