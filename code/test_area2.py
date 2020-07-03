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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def return_null(index, stop):
  if index != stop:
    return False
  else:
    return True


a = range(65001)
import time
start = time.time()
for i in a:
  if not return_null(i, 65000):
    print(i)
    continue
  else:
    print("end found")
    print(time.time() - start)