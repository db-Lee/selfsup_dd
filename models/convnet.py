import torch
import torch.nn as nn
import torch.nn.functional as F

class NoneBlock(nn.Module):
    def __init__(self, num_channels, affine):
        super().__init__()
        
    def forward(self, x):
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            norm_layer(out_channels, affine=True),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        img_shape=(3,32,32),
        num_classes=10,
        num_channels=[128, 256, 512],
        norm="bn"
    ):
        super(ConvNet, self).__init__()

        
        HW = img_shape[1]

        if norm.lower() == "bn":
            norm_layer = nn.BatchNorm2d
        elif norm.lower() == "in":
            norm_layer = nn.InstanceNorm2d
        elif norm.lower() == "none":
            norm_layer = NoneBlock
        else:
            raise NotImplementedError
        
        layers = []
        for i in range(len(num_channels)):
            if i == 0:
                layers.append(ConvBlock(img_shape[0], num_channels[0], norm_layer))
            else:
                layers.append(ConvBlock(num_channels[i-1], num_channels[i], norm_layer))
            HW = HW // 2
        self.layers = nn.ModuleList(layers)
        self.num_features = HW*HW*num_channels[-1]
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def embed(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        return x
"""

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            #nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        x = self.block(x)
        return x

class ConvNet(nn.Module):
    def __init__(
        self,
        img_shape=(3,32,32),
        num_classes=10,
        num_layers=3,
        num_channels=128
    ):
        super(ConvNet, self).__init__()

        layers = []
        HW = img_shape[1]
        for i in range(num_layers):
            if i == 0:
                layers.append(ConvBlock(img_shape[0], num_channels))
            else:
                layers.append(ConvBlock(num_channels, num_channels))
            HW = HW // 2
        self.layers = nn.ModuleList(layers)
        self.num_features = HW*HW*num_channels
        self.fc = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def embed(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)
        return x
"""