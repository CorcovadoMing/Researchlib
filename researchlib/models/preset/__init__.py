import torchvision.models as models
from functools import partial


class Preset(object):
    Alexnet = partial(models.alexnet, pretrained=True, progress=False)
    Resnet18 = partial(models.resnet.resnet18, pretrained=True, progress=False)
    Resnet34 = partial(models.resnet.resnet34, pretrained=True, progress=False)
    Resnet50 = partial(models.resnet.resnet50, pretrained=True, progress=False)
    Resnet101 = partial(models.resnet.resnet101, pretrained=True, progress=False)
    Resnet152 = partial(models.resnet.resnet152, pretrained=True, progress=False)
    
    ResNext50 = partial(models.resnet.resnext50_32x4d, pretrained=True, progress=False)
    ResNext101 = partial(models.resnet.resnext101_32x8d, pretrained=True, progress=False)
    
    WideResnet50 = partial(models.resnet.wide_resnet50_2, pretrained=True, progress=False)
    WideResnet101 = partial(models.resnet.wide_resnet101_2, pretrained=True, progress=False)
    
    Vgg11 = partial(models.vgg.vgg11, pretrained=True, progress=False)
    Vgg11BN = partial(models.vgg.vgg11_bn, pretrained=True, progress=False)
    Vgg13 = partial(models.vgg.vgg13, pretrained=True, progress=False)
    Vgg13BN = partial(models.vgg.vgg13_bn, pretrained=True, progress=False)
    Vgg16 = partial(models.vgg.vgg16, pretrained=True, progress=False)
    Vgg16BN = partial(models.vgg.vgg16_bn, pretrained=True, progress=False)
    Vgg19 = partial(models.vgg.vgg19, pretrained=True, progress=False)
    Vgg19BN = partial(models.vgg.vgg19_bn, pretrained=True, progress=False)
    
    Squeezenet = partial(models.squeezenet.squeezenet1_1, pretrained=True, progress=False)
    InceptionV3 = partial(models.inception.inception_v3, pretrained=True, progress=False)
    
    Densenet121 = partial(models.densenet.densenet121, pretrained=True, progress=False)
    Densenet169 = partial(models.densenet.densenet169, pretrained=True, progress=False)
    Densenet201 = partial(models.densenet.densenet201, pretrained=True, progress=False)
    Densenet161 = partial(models.densenet.densenet161, pretrained=True, progress=False)
    
    MobilenetV2 = partial(models.mobilenet.mobilenet_v2, pretrained=True, progress=False)
    
    MNasnet0_5 = partial(models.mnasnet.mnasnet0_5, pretrained=True, progress=False)
    MNasnet0_75 = partial(models.mnasnet.mnasnet0_75, pretrained=True, progress=False)
    MNasnet1_0 = partial(models.mnasnet.mnasnet1_0, pretrained=True, progress=False)
    MNasnet1_3 = partial(models.mnasnet.mnasnet1_3, pretrained=True, progress=False)
    
    ShufflenetV2x0_5 = partial(models.shufflenetv2.shufflenet_v2_x0_5, pretrained=True, progress=False)
    ShufflenetV2x1_0 = partial(models.shufflenetv2.shufflenet_v2_x1_0, pretrained=True, progress=False)
    ShufflenetV2x1_5 = partial(models.shufflenetv2.shufflenet_v2_x1_5, pretrained=True, progress=False)
    ShufflenetV2x2_0 = partial(models.shufflenetv2.shufflenet_v2_x2_0, pretrained=True, progress=False)