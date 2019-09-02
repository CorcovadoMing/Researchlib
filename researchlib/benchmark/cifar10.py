from ..layers import layer
from ..dataset import *
from ..models import *
from ..runner.preprocessing import *
from ..runner.augmentation import *
from ..blocks.unit import unit

import torch
from torchvision import datasets
from torchvision import transforms

def fetch_bylabel(label):
    if label == 10:
        normalizer = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                          std=[0.2471, 0.2435, 0.2616])
        data_cls = datasets.CIFAR10
    else:
        normalizer = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                          std=[0.2675, 0.2565, 0.2761])
        data_cls = datasets.CIFAR100
    return normalizer, data_cls


def load_dataset(label, batch_size):
    normalizer, data_cls = fetch_bylabel(label)

    train_loader = torch.utils.data.DataLoader(
        data_cls("./data/cifar{}".format(label), train=True, download=True,
                 transform=transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     normalizer
                 ])),
        batch_size=batch_size, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        data_cls("./data/cifar{}".format(label), train=False, download=False,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     normalizer
                 ])),
        batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

class _cifar10:

    def __init__(self, model_name, batch_size=128):
        self.available_models = [
            x for x in _cifar10.__dict__.keys() if type(x) == type
        ]

        self.train_loader = VisionDataset(
            vision.CIFAR10, batch_size=batch_size, train=True)
        self.test_loader = VisionDataset(
            vision.CIFAR10, batch_size=batch_size, train=False)
        
#         self.train_loader, self.test_loader = load_dataset(10, batch_size)

        try:
            getattr(self, model_name)()
        except:
            raise ValueError(
                '{} has not implemented, available models are: {}'.format(
                    model_name, self.available_models))

    def shakedrop_pyramidnet_272(self):
        """ reach 97% accuracy in 300 epochs
        """
        from ..models.auto_conv_net import AutoConvNet
        from ..runner import Runner
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
                filters=(base_dim, -1),
                pool_freq=pool_freq,
                no_end_pool=no_end_pool,
                branch_attention=False,
                shakedrop=True,
                preact=True,
                erased_activator=True,
                filter_policy='pyramid',
                pyramid_alpha=200,
                shortcut='padding'),
            layer.BatchNorm2d(out_dim),
            layer.ReLU(inplace=True),
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
            multigpu=True,
            monitor_state='acc',
            monitor_mode='max',
            swa=False, 
            weight_decay=1e-4)
        self.runner.init_model('xavier_normal')
        print(self.runner.describe())

        self.runner.preprocessing([Normalizer()
                                  ]).augmentation([HFlip(), Crop2d()]).fit(
                                      300,
                                      1e-1,
                                      policy='',
                                      multisteps=[150, 225])
#         self.runner.fit(300, 1e-1, policy='', multisteps=[150, 225])
