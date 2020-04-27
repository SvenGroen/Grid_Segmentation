import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import sigmoid
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class ConvSame_3_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(8, 2, 3, padding=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class Deeplab_Res101(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.classifier = DeepLabHead(2048, 2)
        # self.base.classifier[4] = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.base(x)
        return x["out"]


class Deeplab_Res50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)

    def forward(self, x):
        x = self.base(x)
        return x["out"]

'''
class Deeplab_Mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.mobilenet_v2(pretrained=True)
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.classifier = DeepLabHead(1280, 2)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1280, out_channels=1280, kernel_size=3, stride=36, padding=3),
            # nn.ConvTranspose2d(in_channels=640, out_channels=320, kernel_size=3, stride=2, padding=1),
            # nn.ConvTranspose2d(in_channels=320, out_channels=160, kernel_size=3, stride=2, padding=1),
            # nn.ConvTranspose2d(in_channels=160, out_channels=80, kernel_size=3, stride=2, padding=1),
            # nn.ConvTranspose2d(in_channels=80, out_channels=40, kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        x = self.base.features(x)
        x = self.up(x)
        x = self.base.classifier(x)
        return x["out"]


class Res18_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=True, progress=True)
        self.base = torch.nn.Sequential(*(list(net.children())[:-1]))
        for param in self.base.parameters():
            param.requires_grad = False
        self.out1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=2)
        self.out2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=2, stride=2)
        self.out3 = nn.ConvTranspose2d(in_channels=512, out_channels=2, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        x = self.base(x)
        # x =
        # x = self.out1(x)
        # x = self.out2(x)
        # x = self.out3(x)
        return x
'''
