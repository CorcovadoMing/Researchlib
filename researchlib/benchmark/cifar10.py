from ..runner.preprocessing import *
from ..runner.augmentation import *
from ..dataset import *
from ..blocks import block, unit
from ..layers import layer
import torch
from torchvision import datasets
from torchvision import transforms


class _cifar10:
    def __init__(self, model_name, batch_size = 128, benchmark = False, **kwargs):
        self.available_models = [x for x in _cifar10.__dict__.keys() if type(x) == type]
        self.exp_name = model_name
        self.train_loader = VisionDataset(vision.CIFAR10, batch_size = batch_size, train = True)
        self.test_loader = VisionDataset(vision.CIFAR10, batch_size = batch_size, train = False)
        self.kwargs = kwargs

        try:
            getattr(self, model_name)()
        except:
            raise ValueError(
                '{} has not implemented, available models are: {}'.format(
                    model_name, self.available_models
                )
            )

        if benchmark:
            self.runner.submit_benchmark('Classification', comments = {'comments': model_name})

    def pyramidnet_272(self, shakedrop = False):
        from ..runner import Runner
        from ..models import AutoConvNet, builder, Heads
        input_dim = 3
        base_dim = 16
        num_layer = 272
        pyramid_alpha = 200
        no_end_pool = True

        num_block = (num_layer - 2) // 3  # 1 res-bottleneck has 3 convs
        pool_freq = num_block // 3  # 3 groups
        out_dim = (base_dim + pyramid_alpha) * 4

        self.model = builder([
            # group 1, 2 and 3
            AutoConvNet(
                layer.Conv2d,
                unit.conv,
                input_dim,
                num_block,
                'residual-bottleneck',
                filters = (base_dim, -1),
                pool_freq = pool_freq,
                no_end_pool = no_end_pool,
                branch_attention = False,
                shakedrop = shakedrop,
                preact = True,
                erased_activator = True,
                filter_policy = 'pyramid',
                pyramid_alpha = 200,
                shortcut = 'padding'
            ),
            layer.BatchNorm2d(out_dim),
            layer.ReLU(inplace = True),
            layer.AdaptiveAvgPool2d(1),
            layer.Flatten(),
            layer.Linear(out_dim, 10),
            layer.LogSoftmax(-1)
        ])

        self.runner = Runner(
            self.model,
            self.train_loader,
            self.test_loader,
            'nesterov',
            'nll',
            multigpu = True,
            monitor_state = 'acc',
            monitor_mode = 'max',
            swa = False
        )
        self.runner.init_model('xavier_normal')
        print(self.runner.describe())
        self.runner.start_experiment(self.exp_name)

        self.runner.preprocessing([Normalizer()]).augmentation([HFlip(), Crop2d()]).fit(
            300,
            1e-1,
            policy = 'fixed',
            weight_decay = 1e-4,
            weight_decay_policy = 'fixed',
            multisteps = [150, 225]
        )

    def shakedrop_pyramidnet_272(self):
        """ reach 97% accuracy in 300 epochs with 4 gpus
        """
        self.pyramidnet_272(shakedrop = True)

    def dawnfast(self):
        from ..runner import Runner
        from ..models import AutoConvNet, builder, Heads
        model = builder([
            block.VGGBlock(
                layer.Conv2d, 3, 64, False, True, False, unit = unit.conv, blur = True
            ),
            block.DAWNBlock(
                layer.Conv2d, 64, 128, True, True, False, unit = unit.conv, blur = True
            ),
            block.VGGBlock(
                layer.Conv2d, 128, 256, True, True, False, unit = unit.conv, blur = True
            ),
            block.DAWNBlock(
                layer.Conv2d, 256, 512, True, True, False, unit = unit.conv, blur = True
            ),
            layer.AdaptiveAvgPool2d(1),
            layer.Flatten(),
            layer.Linear(512, 10),
            layer.LogSoftmax(-1)
        ])

        self.runner = Runner(
            model,
            self.train_loader,
            self.test_loader,
            'adamw',
            'smooth_nll',
            lookahead = True,
            monitor_state = 'acc',
            monitor_mode = 'max'
        )
        self.runner.init_model('default')
        self.runner \
            .preprocessing([Normalizer()]) \
            .augmentation([HFlip(), Crop2d(), Cutout()]) \
            .fit(14,
                 1e-2,
                 prefetch=True,
                 plot=False)

    def test(self):
        from ..runner import Runner
        from ..models import AutoConvNet, builder, Heads
        model = builder([
            layer.ManifoldMixup(),
            block.VGGBlock(
                layer.WSConv2d,
                3,
                64,
                False,
                True,
                False,
                unit = unit.conv,
                norm_type = self.get_param('norm_type'),
                bn_affine = self.get_param('bn_affine'),
                gamma_range = self.get_param('gamma_range'),
                beta_range = self.get_param('beta_range'),
                blur = True
            ),
            layer.ManifoldMixup(),
            block.DAWNBlock(
                layer.WSConv2d,
                64,
                128,
                True,
                True,
                False,
                unit = unit.conv,
                norm_type = self.get_param('norm_type'),
                bn_affine = self.get_param('bn_affine'),
                gamma_range = self.get_param('gamma_range'),
                beta_range = self.get_param('beta_range'),
                blur = True
            ),
            layer.ManifoldMixup(),
            block.VGGBlock(
                layer.WSConv2d,
                128,
                256,
                True,
                True,
                False,
                unit = unit.conv,
                norm_type = self.get_param('norm_type'),
                bn_affine = self.get_param('bn_affine'),
                gamma_range = self.get_param('gamma_range'),
                beta_range = self.get_param('beta_range'),
                blur = True
            ),
            layer.ManifoldMixup(),
            block.DAWNBlock(
                layer.WSConv2d,
                256,
                512,
                True,
                True,
                False,
                unit = unit.conv,
                norm_type = self.get_param('norm_type'),
                bn_affine = self.get_param('bn_affine'),
                gamma_range = self.get_param('gamma_range'),
                beta_range = self.get_param('beta_range'),
                blur = True
            ),
            layer.ManifoldMixup(),
            layer.AdaptiveAvgPool2d(1),
            layer.Flatten(),
            layer.Linear(512, 10),
            layer.LogSoftmax(-1)
        ])

        self.runner = Runner(
            model,
            self.train_loader,
            self.test_loader,
            'adamw',
            'smooth_nll',
            lookahead = True,
            monitor_state = 'acc',
            monitor_mode = 'max'
        )
        self.runner.init_model('default')
        self.runner \
            .preprocessing([Normalizer()]) \
            .augmentation([]) \
            .fit(self.kwargs['epoches'], weight_decay=1e-4, weight_decay_policy='fixed',
                 mmixup_alpha=self.get_param('mmixup_alpha'),
                 random_mmixup=self.get_param('random_mmixup'),
                 fixed_mmixup=self.get_param('fixed_mmixup'),
                 prefetch=False,
                 plot=False)

    def get_param(self, key, default_value = None):
        return self.kwargs[key] if key in self.kwargs else default_value
