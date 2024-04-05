import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

from models.convnet import ConvNet
from models.resnet import ResNet10, ResNet18
from models.vgg import VGG11
from models.alexnet import AlexNet
from models.mobilenet import MobileNet

def get_model(name, img_shape=(3,32,32), num_classes=10, dropout=0.0):
    if "convnet" in name.lower():
        name_splited = name.lower().split("_")
        num_channels = []
        for idx, ns in enumerate(name_splited):
            if idx != 0:
                if ns.isdigit():
                    num_channels.append(int(ns))
                else:
                    norm = ns
        model = ConvNet(img_shape, num_classes, num_channels, norm)
    elif "resnet10" == name.lower():
        model = ResNet10(num_classes)
    elif "resnet18" == name.lower():
        model = ResNet18(num_classes)
    elif "vgg" == name.lower():
        model = VGG11(img_shape[1], num_classes)
    elif "alexnet" == name.lower():
        model = AlexNet(img_shape[1], num_classes)
    elif "mobilenet" == name.lower():
        model = MobileNet(num_classes)
    else:
        raise NotImplementedError
    if dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            model.fc
        )
    return model