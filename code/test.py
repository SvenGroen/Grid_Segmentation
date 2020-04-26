import torch
import torchvision
import torch.nn.functional as F
from models.custom.simple_models.simple_models import *

net = torchvision.models.resnet18()
net2 = torchvision.models.segmentation.deeplabv3_resnet50()
# print(net)

newmodel = torch.nn.Sequential(*(list(net.children())[:-2]))
print(newmodel)
print("END OF PYTHON FILE")


