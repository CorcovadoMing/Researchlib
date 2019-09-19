class SaveFeatures:
    def __init__(self, m):
        self.features = None
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()
