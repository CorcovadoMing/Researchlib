class BGR2RGB:
    def __init__(self):
        pass
        
    def __call__(self, x):
        return x[..., (2,1,0)]