class Matrix:
    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        pass
    
    def output(self):
        pass
    
    def reset(self):
        pass
        
    def prepare(self, y_pred, y_true):
        y_pred, y_true = y_pred.cpu(), y_true.cpu()
        if y_pred.shape != y_true.shape:
            y_pred = y_pred.argmax(-1)
        return y_pred, y_true