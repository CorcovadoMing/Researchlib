import torchvision.models as models
from functools import partial
from .efficientnet import EfficientNet


def load_efficientnet(name, pretrained):
    if pretrained:
        return EfficientNet.from_pretrained(name)
    else:
        return EfficientNet.from_name(name)
    

class ImageNet(object):
    AlexNet = partial(models.alexnet, pretrained=True, progress=False)
    ResNet18 = partial(models.resnet.resnet18, pretrained=True, progress=False)
    ResNet34 = partial(models.resnet.resnet34, pretrained=True, progress=False)
    ResNet50 = partial(models.resnet.resnet50, pretrained=True, progress=False)
    ResNet101 = partial(models.resnet.resnet101, pretrained=True, progress=False)
    ResNet152 = partial(models.resnet.resnet152, pretrained=True, progress=False)
    
    ResNext50 = partial(models.resnet.resnext50_32x4d, pretrained=True, progress=False)
    ResNext101 = partial(models.resnet.resnext101_32x8d, pretrained=True, progress=False)
    
    WideResNet50 = partial(models.resnet.wide_resnet50_2, pretrained=True, progress=False)
    WideResNet101 = partial(models.resnet.wide_resnet101_2, pretrained=True, progress=False)
    
    Vgg11 = partial(models.vgg.vgg11, pretrained=True, progress=False)
    Vgg11BN = partial(models.vgg.vgg11_bn, pretrained=True, progress=False)
    Vgg13 = partial(models.vgg.vgg13, pretrained=True, progress=False)
    Vgg13BN = partial(models.vgg.vgg13_bn, pretrained=True, progress=False)
    Vgg16 = partial(models.vgg.vgg16, pretrained=True, progress=False)
    Vgg16BN = partial(models.vgg.vgg16_bn, pretrained=True, progress=False)
    Vgg19 = partial(models.vgg.vgg19, pretrained=True, progress=False)
    Vgg19BN = partial(models.vgg.vgg19_bn, pretrained=True, progress=False)
    
    SqueezeNet = partial(models.squeezenet.squeezenet1_1, pretrained=True, progress=False)
    InceptionV3 = partial(models.inception.inception_v3, pretrained=True, progress=False)
    
    DenseNet121 = partial(models.densenet.densenet121, pretrained=True, progress=False)
    DenseNet169 = partial(models.densenet.densenet169, pretrained=True, progress=False)
    DenseNet201 = partial(models.densenet.densenet201, pretrained=True, progress=False)
    DenseNet161 = partial(models.densenet.densenet161, pretrained=True, progress=False)
    
    MobileNetV2 = partial(models.mobilenet.mobilenet_v2, pretrained=True, progress=False)
    
    MNasNet0_5 = partial(models.mnasnet.mnasnet0_5, pretrained=True, progress=False)
    MNasNet0_75 = partial(models.mnasnet.mnasnet0_75, pretrained=True, progress=False)
    MNasNet1_0 = partial(models.mnasnet.mnasnet1_0, pretrained=True, progress=False)
    MNasNet1_3 = partial(models.mnasnet.mnasnet1_3, pretrained=True, progress=False)
    
    ShuffleNetV2x0_5 = partial(models.shufflenetv2.shufflenet_v2_x0_5, pretrained=True, progress=False)
    ShuffleNetV2x1_0 = partial(models.shufflenetv2.shufflenet_v2_x1_0, pretrained=True, progress=False)
    ShuffleNetV2x1_5 = partial(models.shufflenetv2.shufflenet_v2_x1_5, pretrained=True, progress=False)
    ShuffleNetV2x2_0 = partial(models.shufflenetv2.shufflenet_v2_x2_0, pretrained=True, progress=False)
    
    EfficientNetB0 = partial(load_efficientnet, 'efficientnet-b0', pretrained=True)
    EfficientNetB1 = partial(load_efficientnet, 'efficientnet-b1', pretrained=True)
    EfficientNetB2 = partial(load_efficientnet, 'efficientnet-b2', pretrained=True)
    EfficientNetB3 = partial(load_efficientnet, 'efficientnet-b3', pretrained=True)
    EfficientNetB4 = partial(load_efficientnet, 'efficientnet-b4', pretrained=True)
    EfficientNetB5 = partial(load_efficientnet, 'efficientnet-b5', pretrained=True)
    EfficientNetB6 = partial(load_efficientnet, 'efficientnet-b6', pretrained=True)
    EfficientNetB7 = partial(load_efficientnet, 'efficientnet-b7', pretrained=True)