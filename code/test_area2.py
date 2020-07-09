import os
import time
import datetime
from pathlib import Path

import cv2
import yaml
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataLoader.Datasets.Examples_Green.NY.NY_mixed import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from models.custom.simple_models.simple_models import *
from models.DeepLabV3PlusPytorch.network import *
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.tensor([1])
b = torch.tensor([2])
c = torch.tensor([3])

old = [None, None]
old[1] = a
old[0] = b

old = list(np.flip(old))
print(type(old))
c =  old + [c]
d = torch.cat(c, dim=0)
e = torch.stack(c,dim=0)
print(c)
