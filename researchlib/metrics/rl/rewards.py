class ExtrinsicRewards:
    def __init__(self):
        super().__init__()
    
    def __call__(self, eps_trajection):
        l = [sum(trajection['reward']) for trajection in eps_trajection]
        return sum(l)/len(l)
    
    
class IntrinsicRewards:
    def __init__(self):
        super().__init__()
    
    def __call__(self, eps_trajection):
        l = [sum(trajection['intrinsic']) / len(trajection['intrinsic']) for trajection in eps_trajection]
        return sum(l)/len(l)