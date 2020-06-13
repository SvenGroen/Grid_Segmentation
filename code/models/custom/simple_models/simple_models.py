import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import sigmoid
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from models.DeepLabV3PlusPytorch.network import *
from utils.convGRU import *
from utils.convlstm import *


class ConvSame_3_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(8, 2, 3, padding=1, stride=1)

    def forward(self, x, *args):
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

    def forward(self, x, *args):
        x = self.base(x)
        return x["out"]

class Deeplab_Res101V2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True)
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.classifier = DeepLabHead(2048, 2)
        # self.base.classifier[4] = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1, stride=1)

    def forward(self, x, *args):
        x = self.base(x)
        return x["out"]


class Deeplab_Res50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=2)
        self.backbone = self.base.backbone
        self.classifier = self.base.classifier

    def forward(self, x, *args):
        x = self.base(x)
        # x = self.backbone(x)["out"]
        # x= self.classifier(x)
        return x["out"]


class FCN_Res50(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=2)

    def forward(self, x, *args):
        x = self.base(x)
        return x["out"]

class Deeplabv3Plus_mobile(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
    def forward(self, x, *args):
        return self.base(x)

class Deep_mobile_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        self.backbone = self.base.backbone
        self.classifier = self.base.classifier
        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=False)
        self.hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        out, self.hidden = self.lstm(out, self.hidden)
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        return out[-1].squeeze(1)

class Deep_mobile_lstmV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        self.backbone = self.base.backbone
        self.classifier = self.base.classifier
        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=False)
        self.lstmcell = ConvLSTMCell(input_dim= 2, hidden_dim=2, kernel_size=(3,3), bias=True)
        self.hidden = None

    def forward(self, x, hidden_state, *args):
        if len(args) != 0:
            old_pred = args[0]
            for i in range(len(old_pred)):
                if old_pred[i] is not None and len(old_pred[i].shape) != len(x.shape) + 1:
                    old_pred[i] = old_pred[i].unsqueeze(1)
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        if len(args) != 0:
            if None in old_pred:
                for i in range(len(old_pred)):
                    old_pred[i] = torch.zeros_like(out)
            out = [out] + old_pred
            out = torch.cat(out, dim =1)

        out, self.hidden = self.lstm(out, self.hidden)
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        # out = F.interpolate(out[-1].squeeze(1), size=input_shape, mode='bilinear', align_corners=False)
        out = out[0][:,-1,:,:,:]
        return out


class Deep_mobile_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        self.backbone = self.base.backbone
        self.classifier = self.base.classifier
        self.gru = ConvGRU(input_size=(270, 512), input_dim=2, hidden_dim=[64], kernel_size=(3, 3), num_layers=1,
                            dtype=torch.FloatTensor, batch_first=True, bias= True, return_all_layers=True)

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = x.unsqueeze(1)
        x = self.gru(x)
        return x[-1]


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
