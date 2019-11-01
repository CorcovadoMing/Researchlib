class _pack:
    def __init__(self, value):
        self.value = value
        
def _unpack(obj):
    if type(obj) == _pack:
        return obj.value
    else:
        return obj