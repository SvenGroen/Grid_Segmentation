import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import sigmoid

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