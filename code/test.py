import torch
import torchvision
import torch.nn.functional as F
from models.custom.simple_models.simple_models import *
import numpy as np
import matplotlib.pyplot as plt
net = torchvision.models.resnet18()
net2 = torchvision.models.segmentation.deeplabv3_resnet50()
# print(net)

newmodel = torch.nn.Sequential(*(list(net.children())[:-2]))

mobile = torchvision.models.mobilenet_v2(pretrained=True)
# print(mobile)
a =np.array([34,64])
b= [2,4,7,9,12,34,67,89,99]



from models.DeepLabV3PlusPytorch.network import *

net = modeling.deeplabv3_mobilenet()
print(net)