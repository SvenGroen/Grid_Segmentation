import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import sigmoid
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from models.DeepLabV3PlusPytorch.network import *
from models.DeepLabV3PlusPytorch.network._deeplab import DeepLabHeadV3PlusLSTM, DeepLabHeadV3PlusGRU

from utils.convGRU import *
from utils.convlstm import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# BASE
class Deeplabv3Plus_base(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

    def forward(self, x, *args):
        return self.base(x)


# --- LSTMs ---
class Deeplabv3Plus_lstmV1(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=False)
        self.hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        out, self.hidden = self.lstm(out, self.hidden)
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        return out[-1].squeeze(1)


class Deeplabv3Plus_lstmV2(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.lstm = ConvLSTM(input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=False)
        self.hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        x = self.base(x)
        out = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        if len(args) != 0:
            old_pred = args[0]
            # initialize if necessary
            if None in old_pred:
                for i in range(len(old_pred)):
                    old_pred[i] = torch.zeros_like(out)
            # match shape
            elif len(old_pred[0].shape) != len(out.shape):
                for i in range(len(old_pred)):
                    old_pred[i] = old_pred[i].unsqueeze(1)
            out = old_pred + [out]
            out = torch.cat(out, dim=1)

        out, self.hidden = self.lstm(out, self.hidden)
        out = out[0][:, 0, :, :, :]  # <--- not to sure if 0 or -1
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        return out


class Deeplabv3Plus_lstmV3(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256

        self.backbone = self.base.backbone
        self.classifier = DeepLabHeadV3PlusLSTM(in_channels, low_level_channels, 2, [12, 24, 36])
        self.hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out


# --- GRU ---
class Deeplabv3Plus_gruV1(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.gru = ConvGRU(input_size=(270, 512), input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1,
                           dtype=torch.FloatTensor, batch_first=True, bias=True, return_all_layers=True)
        self.hidden = [None]

    def forward(self, x, *args):
        x = self.base(x)
        x = x.unsqueeze(1)
        out, self.hidden = self.gru(x, self.hidden[-1])
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        out = out[0][:, 0, :, :, :]  # <--- not to sure if 0 or -1
        return out


class Deeplabv3Plus_gruV2(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)

        self.gru = ConvGRU(input_size=(270, 512), input_dim=2, hidden_dim=[2], kernel_size=(3, 3), num_layers=1,
                           dtype=torch.FloatTensor, batch_first=True, bias=True, return_all_layers=True)
        self.hidden = [None]

    def forward(self, x, *args):
        out = self.base(x)
        out = out.unsqueeze(1)  # add "timestep" dimension

        if len(args) != 0:
            old_pred = args[0]
            # initialize if necessary
            if None in old_pred:
                for i in range(len(old_pred)):
                    old_pred[i] = torch.zeros_like(out)
            # match shape
            elif len(old_pred[0].shape) != len(out.shape):
                for i in range(len(old_pred)):
                    old_pred[i] = old_pred[i].unsqueeze(1)  # add "timestep" dimension
            out = old_pred + [out]
            out = torch.cat(out, dim=1)
        out, self.hidden = self.gru(out, self.hidden[-1])
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        out = out[0][:, 0, :, :, :]  # <--- not to sure if 0 or -1
        return out


class Deeplabv3Plus_gruV3(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=2, pretrained_backbone=True)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=2, pretrained_backbone=True)
            in_channels = 2048
            low_level_channels = 256

        self.backbone = self.base.backbone
        self.classifier = DeepLabHeadV3PlusGRU(in_channels, low_level_channels, 2, [12, 24, 36])
        self.hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        out = self.classifier(features)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        return out

# ----------- other models ------------------
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

class Deeplabv3Plus_rgb(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=3, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=3, pretrained_backbone=True)

    def forward(self, x, *args):
        return self.base(x)

class Deeplabv3Plus_rgb_gruV1(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=3, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=3, pretrained_backbone=True)

        self.gru = ConvGRU(input_size=(270, 512), input_dim=3, hidden_dim=[3], kernel_size=(3, 3), num_layers=1,
                           dtype=torch.FloatTensor, batch_first=True, bias=True, return_all_layers=True)
        self.hidden = [None]

    def forward(self, x, *args):
        x = self.base(x)
        x = x.unsqueeze(1)
        out, self.hidden = self.gru(x, self.hidden[-1])
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        out = out[0][:, 0, :, :, :]  # <--- not to sure if 0 or -1
        return out

class Deeplabv3Plus_rgb_lstmV1(nn.Module):
    def __init__(self, backbone="mobilenet"):
        super().__init__()
        if backbone == "mobilenet":
            self.base = deeplabv3plus_mobilenet(num_classes=3, pretrained_backbone=True)
        elif backbone == "resnet50":
            self.base = deeplabv3plus_resnet50(num_classes=3, pretrained_backbone=True)

        self.lstm = ConvLSTM(input_dim=3, hidden_dim=[3], kernel_size=(3, 3), num_layers=1, batch_first=True,
                             bias=True,
                             return_all_layers=False)
        self.hidden = None

    def forward(self, x, *args):
        input_shape = x.shape[-2:]
        out = self.base(x)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)
        out = out.unsqueeze(1)
        out, self.hidden = self.lstm(out, self.hidden)
        self.hidden = [tuple(state.detach() for state in i) for i in self.hidden]
        return out[-1].squeeze(1)