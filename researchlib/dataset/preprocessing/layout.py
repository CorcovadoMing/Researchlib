class Layout:
    def __init__(self, *layout_format):
        self.source, self.target = layout_format
    
    def __call__(self, x):
        return x.transpose([self.source.index(d) for d in self.target])
