#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F


# refer to https://github.com/CoinCheung/Deeplab-Large-FOV/blob/master/lib/model.py
class DeepLab_v1(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super(DeepLab_v1, self).__init__(*args, **kwargs)
        # vgg16 = torchvision.models.vgg16()

        layers = []
        layers.append(nn.Conv2d(in_dim, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 2, padding = 1))

        layers.append(nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))

        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(512,
            512,
            kernel_size = 3,
            stride = 1,
            padding = 2,
            dilation = 2))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.MaxPool2d(3, stride = 1, padding = 1))
        self.features = nn.Sequential(*layers)

        classifier = []
        classifier.append(nn.AvgPool2d(3, stride = 1, padding = 1))
        classifier.append(nn.Conv2d(512,
            1024,
            kernel_size = 3,
            stride = 1,
            padding = 4,
            dilation = 4))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0))
        classifier.append(nn.ReLU(inplace=True))
        classifier.append(nn.Dropout(p=0.5))
        classifier.append(nn.Conv2d(1024, out_dim, kernel_size=1))
        self.classifier = nn.Sequential(*classifier)

        self.init_weights()


    def forward(self, x):
        N, C, H, W = x.size()
        x = self.features(x)
        x = self.classifier(x)
        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
        return x

    def init_weights(self):
        vgg = torchvision.models.vgg16(pretrained=False)
        state_vgg = vgg.features.state_dict()
        self.features.load_state_dict(state_vgg)

        for ly in self.classifier.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                nn.init.constant_(ly.bias, 0)


if __name__ == "__main__":
    net = DeepLabLargeFOV(3, 2)
    in_ten = torch.randn(1, 3, 224, 224)
    print(in_ten.size())
    out = net(in_ten)
    print(out.size())