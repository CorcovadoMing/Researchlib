import time


class Timer:
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations
        self.start_time = 0
        self.cur_iterations = 0
        self.cur_time = 0
    
    def clear(self):
        self.start_time = time.time()
        self.cur_iterations = 0
        self.cur_time = 0
        
    def _tick(self):
        self.cur_iterations += 1
        self.cur_time = time.time()
    
    def output(self):
        self._tick()
        accum = self.cur_time - self.start_time
        eta = ((accum / self.cur_iterations) * self.max_iterations) - accum
        return f'Iterations: ({self.cur_iterations}/{self.max_iterations}), ETA: {accum:.1f}:{eta:.1f} secs', self.cur_iterations/self.max_iterations