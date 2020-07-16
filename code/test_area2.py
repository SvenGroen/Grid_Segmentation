import json
import argparse
import time
import sys

import cv2
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
# import models and dataset
from utils.metrics import *
from models.custom.simple_models.simple_models import *
from models.custom.simple_models.UNet import *
from models.DeepLabV3PlusPytorch.network import *
from models.ICNet.models import ICNet
from models.ICNet.utils import ICNetLoss, IterationPolyLR, SegmentationMetric, SetupLogger
from DataLoader.Datasets.Youtube.Youtube_Greenscreen import *
from DataLoader.Datasets.Youtube.Youtube_Greenscreen_mini import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metrics = defaultdict(AverageMeter)
metrics["IoU"].update(0)
metrics["IoU"].update(1)
metric_log = list()
metric_log.append(metrics)
metrics2 = defaultdict(AverageMeter)
metrics2["IoU"].update(2)
metrics2["IoU"].update(3)
metric_log.append(metrics2)

print("Save:")
checkpoint = {}
checkpoint["metric_log"] = metric_log
torch.save(checkpoint, "test.pth.tar")


checkpoint2 = torch.load("test.pth.tar")

for k in checkpoint2:
    print(k)