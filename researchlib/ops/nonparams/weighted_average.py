class _WeightedAverage:
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, *x):
        accum = 0
        count = 0
        for i, j in zip(x, self.weight):
            if i is not None:
                accum += i * j
                count += j
        return accum / count
