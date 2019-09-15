class ParameterManager:
    keys_whitelist = []
    buffer = {}
    params = {}

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

    def get_param(self,
                  key,
                  init_value=None,
                  required=False,
                  validator=lambda _: True):
        # register key
        if key not in ParameterManager.keys_whitelist:
            ParameterManager.keys_whitelist.append(key)

        if required and key not in self.kwargs:
            raise ValueError("{} is required in **kwargs".format(key))

        query = None
        if key in self.kwargs:
            query = self.kwargs[key]
        else:
            query = init_value

        if query is not None and not validator(query):
            raise ValueError('{} is not a proper value for key:{}'.format(
                query, key))

        ParameterManager.params[key] = query
        return query

    @classmethod
    def save_buffer(cls, key, value):
        ParameterManager.buffer[key] = value

    @classmethod
    def get_buffer(cls, key, clear=True):
        if key not in ParameterManager.buffer:
            raise ValueError("Key {} is not in buffer".format(key))
        result = ParameterManager.buffer[key]
        if clear:
            del ParameterManager.buffer[key]
        return result
