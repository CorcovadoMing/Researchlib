from ..dataset import loader, Augmentations
from ..blocks import block, unit
from ..ops import op
from ..runner import Runner
from ..models import AutoConvNet, Builder, Heads
from ..loss import Loss
from ..metrics import Metrics
from .uploader import uploader


class _cifar10:
    def __init__(self, model_name, batch_size = 128, upload = False, **kwargs):
        self.available_models = [x for x in _cifar10.__dict__.keys() if type(x) == type]
        self.exp_name = model_name
        self.train_loader = loader.TorchDataset('cifar10', True)
        self.test_loader = loader.TorchDataset('cifar10', False)
        self.kwargs = kwargs
        self.runner = None

        try:
            _ = getattr(self, model_name)()
        except:
            raise ValueError(
                '{} has not implemented, available models are: {}'.format(
                    model_name, self.available_models
                )
            )

        if upload:
            if self.runner is None:
                raise ValueError("Runner didn't run yet")
            _uploader = uploader()
            _uploader.submit(self.runner, 'Classification', comments = {'comments': model_name})

    def pyramidnet_272(self, shakedrop = False):
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
                op.Conv2d,
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
            op.BatchNorm2d(out_dim),
            op.ReLU(inplace = True),
            op.AdaptiveAvgPool2d(1),
            op.Flatten(),
            op.Linear(out_dim, 10),
            op.LogSoftmax(-1)
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
        model = Builder.Graph({
            'l2': (AutoConvNet(op.Conv2d, unit.conv, 3, 3, stem={'whitening': 1},
                        type={'order':['dawn', 'vgg'], 'type':'alternative'}, 
                        filters=(64, 512), activator_type='CELU', prepool=True, freeze_scale=True, norm_type=op.GhostBatchNorm2d), 
                   ['x']),
            'l3': (Heads(10, reduce_type='avg'), ['l2']),
            'l4': (op.Multiply(1/2), ['l3']),
            'out': (op.LogSoftmax(-1), ['l4']),
            'loss': (Loss.SmoothNLL(), ['out', 'y']),

            '*x_flip': (op.Flip, ['x']),
            '*shared': (['l2', 'l3', 'l4'], ['x_flip']),
            '*out_tta': (op.LogSoftmax(-1), ['shared']),
            'out_avg': (op.Average, ['out', 'out_tta']),

            'categorical': (Metrics.Categorical(), ['out_avg']),
            'acc': (Metrics.Acc(), ['categorical', 'y'])
        })

        self.runner = Runner(model, 
                             self.train_loader, 
                             self.test_loader, 
                             'nag', 
                             'loss', 
                             output_node='out', 
                             monitor_mode='max', 
                             monitor_state='acc')
        self.runner\
              .normalize('static', 
                         (125.31, 122.95, 113.87), 
                         (62.99, 62.09, 66.70))\
              .preloop()\
              .augmentation([Augmentations.HFlip(), 
                             Augmentations.Crop(32, 32, 4)])\
              .fit(10, 
                   1, 
                   warmup=2, 
                   flatten=2, 
                   flatten_lr=0.1, 
                   fp16=True, 
                   ema_freq=5, 
                   bias_scale=64, 
                   monitor=['acc'])
