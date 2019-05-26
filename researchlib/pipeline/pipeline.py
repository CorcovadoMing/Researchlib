import torch

class Pipe:
    def __init__(self, runners, process):
        self.runners = runners
        self.process = process
    
    @classmethod
    def Parallel(cls, runners):
        return cls(runners, 'parallel')
        
    @classmethod
    def Chain(cls, runners):
        return cls(runners, 'chain')
    
    #------------------------------------------------------------
    
    def fit(self, epochs):
        if self.process == 'parallel':
            for _ in range(epochs):
                for i, runner in enumerate(self.runners):
                    runner.fit(1, _id= 'runner' + str(i+1))
    
    def ensemble(self, x):
        if self.process != 'parallel':
            print('Only parallel pipline supports ensemble')
            return
            
        result = None
        for runner in self.runners:
            runner.model.eval()
            if result is None:
                result = runner.model(x)
            else:
                result += runner.model(x)
        return result / len(self.runners)