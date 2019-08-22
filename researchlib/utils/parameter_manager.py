class ParameterManager:
    keys_whitelist = []
    
    @classmethod
    def verify_kwargs(cls, **kwargs):
        """must call this after all keys get registered
        """
        for key in kwargs:
            if key not in ParameterManager.keys_whitelist:
                raise ValueError(
                    "'{}' is not allowed. Keys Whitelist: {}".format(
                        key, ParameterManager.keys_whitelist))
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_param(self, key, init_value=None, required=False):
        # register key
        if key not in ParameterManager.keys_whitelist:
            ParameterManager.keys_whitelist.append(key)
            
        if required and key not in self.kwargs:
            raise ValueError("{} is required in **kwargs".format(key))
        try:
            query = self.kwargs[key]
            return query
        except:
            return init_value