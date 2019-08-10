from torch import nn


class _EqualizeLrHelper:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_equallr_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_equallr_orig',
                                  nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def _EqualizeLr(module, name='weight'):
    _EqualizeLrHelper.apply(module, name)
    return module
