class Callback:
    def __init__(self):
        pass
    
    def on_epoch_begin(self, **kwargs):
        '''
            Before each epoch start
        '''
        pass
    
    def on_epoch_end(self, **kwargs):
        '''
            End of epoch
        '''
        pass
    
    def on_iteration_begin(self, **kwargs):
        '''
            Before each minibatch start
        '''
        pass
    
    def on_iteration_end(self, **kwargs):
        '''
            End of each minibatch
        '''
        pass
    
    def on_validation_end(self, **kwargs):
        '''
            After the validation is end
        '''
        pass
    
    def on_update_begin(self, **kwargs):
        '''
            Before the optimzier is applied
        '''
        pass
    
    def on_update_end(self, **kwargs):
        '''
            After the optimzier is applied
        '''
        pass