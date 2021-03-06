class Callback:
    def __init__(self):
        pass

    def on_epoch_begin(self, **kwargs):
        '''
            Before each epoch start
        '''
        return kwargs

    def on_epoch_end(self, **kwargs):
        '''
            End of epoch
        '''
        return kwargs

    def on_iteration_begin(self, **kwargs):
        '''
            Before each minibatch start
        '''
        return kwargs

    def on_iteration_end(self, **kwargs):
        '''
            End of each minibatch
        '''
        return kwargs

    def on_validation_begin(self, **kwargs):
        '''
            Before validation start
        '''
        return kwargs

    def on_validation_end(self, **kwargs):
        '''
            After the validation is end
        '''
        return kwargs

    def on_update_begin(self, **kwargs):
        '''
            Before the optimzier is applied
        '''
        return kwargs

    def on_update_end(self, **kwargs):
        '''
            After the optimzier is applied
        '''
        return kwargs
